CREATE OR REPLACE PROCEDURE retrain(table_name TEXT, test_table TEXT, model_name TEXT, task VARCHAR(20), args_name TEXT)
LANGUAGE plpython3u AS $$
    """
    Retrain a base model using given table
    """
    import base64
    import datetime
    import dill
    from functools import partial
    import re
    import json
    import numpy as np
    import os
    import pandas as pd
    import time
    import torch
    from torch import nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as transforms
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoModelForImageClassification
    from transformers import AutoTokenizer, AutoImageProcessor
    from transformers import ViTForImageClassification, ViTFeatureExtractor
    from tqdm import tqdm

    import sys
    
    sys.path.insert(-1,'')

    from NLP_dataset import NLPDataset
    from CV_dataset import CVDataset
    from test import test_cv, test_nlp

    NLP_TASK = ['SentimentClsCH', 'SentimentClsEN', 'TextClass']
    CV_TASK = ['ImageClass2','ImageClass3','Digit', 'ImageClass']
    TAB_TASK = ['TabClass']

    seed = 42
    torch.manual_seed(seed)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    start_time = time.time()

    query = "SELECT * FROM PRE_MODELS WHERE name='{}'".format(model_name)
    model_path = plpy.execute(query)[0]['path']

    query = "SELECT * FROM args_sync WHERE name='{}'".format(args_name)
    args = plpy.execute(query)[0]

    if task in TAB_TASK:
        query = "SELECT ( \
                SELECT STRING_AGG(CONCAT(key, ': ', value), ',') \
                FROM jsonb_each_text(to_jsonb({}) - 'label') \
                ) AS text, label FROM {}".format(table_name, table_name)
        query2 = "SELECT ( \
            SELECT STRING_AGG(CONCAT(key, ': ', value), ',') \
            FROM jsonb_each_text(to_jsonb({}) - 'label') \
            ) AS text, label, userid FROM {} WHERE userid IN (SELECT DISTINCT userid FROM {}) ".format(test_table, test_table, table_name);
        query3 = "SELECT ( \
            SELECT STRING_AGG(CONCAT(key, ': ', value), ',') \
            FROM jsonb_each_text(to_jsonb({}) - 'label') \
            ) AS text, label, userid FROM {} WHERE userid NOT IN (SELECT DISTINCT userid FROM {}) ".format(test_table, test_table, table_name);
    if task in NLP_TASK:
        query = "SELECT * FROM {}".format(table_name);
        query2 = "SELECT * FROM {} WHERE label IN (SELECT DISTINCT label FROM {}) ".format(test_table, table_name);
        query3 = "SELECT * FROM {} WHERE label NOT IN (SELECT DISTINCT label FROM {}) ".format(test_table, table_name);
    if task in CV_TASK:
        query = "SELECT encode(image, 'base64') AS image, label FROM {}".format(table_name);
        query2 = "SELECT encode(image, 'base64') AS image, label FROM {} WHERE label IN (SELECT DISTINCT label FROM {})".format(test_table, table_name);
        query3 = "SELECT encode(image, 'base64') AS image, label FROM {} WHERE label NOT IN (SELECT DISTINCT label FROM {})".format(test_table, table_name);

    raw_data = plpy.execute(query)
    raw_test_data = plpy.execute(query2)
    raw_test_unlearn_data = plpy.execute(query3)

    data_list = []
    label_list = []
    test_data_list = []
    test_label_list = []
    test_unlearn_data_list = []
    test_unlearn_label_list = []

    if task in NLP_TASK or task in TAB_TASK:
        for item in raw_data:
            data_list.append(item['text'])
            label_list.append(item['label'])
        del raw_data
        for item in raw_test_data:
            test_data_list.append(item['text'])
            test_label_list.append(item['label'])
        del raw_test_data
        for item in raw_test_unlearn_data:
            test_unlearn_data_list.append(item['text'])
            test_unlearn_label_list.append(item['label'])
        del raw_test_unlearn_data
    if task in CV_TASK:
        for item in raw_data:
            data_list.append(base64.b64decode(item['image']))
            label_list.append(item['label'])
        del raw_data
        for item in raw_test_data:
            test_data_list.append(base64.b64decode(item['image']))
            test_label_list.append(item['label'])
        del raw_test_data
        for item in raw_test_unlearn_data:
            test_unlearn_data_list.append(base64.b64decode(item['image']))
            test_unlearn_label_list.append(item['label'])
        del raw_test_unlearn_data

    num_label = max(label_list) + 1

    model = None
    config = AutoConfig.from_pretrained(model_path, num_labels=num_label)
    if task in NLP_TASK or task in TAB_TASK:
        model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config, ignore_mismatched_sizes=True)
    if task in CV_TASK:
        model = AutoModelForImageClassification.from_pretrained(model_path, config=config, ignore_mismatched_sizes=True)
    model.to(device)

    dataloader = None
    if task in NLP_TASK or task in TAB_TASK:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        dataset = NLPDataset(texts=data_list, labels=label_list, tokenizer=tokenizer, max_len=256)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        test_dataset = NLPDataset(texts=test_data_list, labels=test_label_list, tokenizer=tokenizer, max_len=256)
        test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        test_unlearn_dataset = NLPDataset(texts=test_unlearn_data_list, labels=test_unlearn_label_list, tokenizer=tokenizer, max_len=256)
        test_unlearn_dataloader = DataLoader(test_unlearn_dataset, batch_size=16, shuffle=False)
    if task in CV_TASK:
        processor = AutoImageProcessor.from_pretrained(model_path)

        dataset = CVDataset(images=data_list, labels=label_list, processor=processor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        del data_list
        test_dataset = CVDataset(images=test_data_list, labels=test_label_list, processor=processor)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        del test_data_list
        test_unlearn_dataset = CVDataset(images=test_unlearn_data_list, labels=test_unlearn_label_list, processor=processor)
        test_unlearn_dataloader = DataLoader(test_unlearn_dataset, batch_size=32, shuffle=False)
        del test_unlearn_data_list
    
    lr = args['learn_rate']
    weight_decay = args['learn_weight_decay']
    num_epoch = int(args['learn_epoch'])
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    preprocess_time = time.time() - start_time
    step = 0
    total_training_time = 0
    total_test_time = 0
    accuracy_log = []
    if task in NLP_TASK or task in TAB_TASK:
        test_interval_steps = 400

        unlearn_acc = test_nlp(model, test_unlearn_dataloader)
        retain_acc = test_nlp(model, test_dataloader)
        log_entry = {
            "step": 0,
            "training_time": preprocess_time,
            "unlearn_accuracy": unlearn_acc,
            "retain_accuracy": retain_acc
        }
        accuracy_log.append(log_entry)
        print(f"Logged accuracy: {log_entry}")

        model.train()
        start_time = time.time()
        for epoch in range(num_epoch):
            epoch_loss = 0
            for batch in dataloader:
                step += 1
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

                if step % test_interval_steps == 0:
                    test_start_time = time.time()

                    elapsed_training_time = time.time() - start_time - total_test_time
                    unlearn_acc = test_nlp(model, test_unlearn_dataloader)
                    retain_acc = test_nlp(model, test_dataloader)
                    log_entry = {
                        "step": step,
                        "training_time": preprocess_time + elapsed_training_time,
                        "unlearn_accuracy": unlearn_acc,
                        "retain_accuracy": retain_acc
                    }
                    accuracy_log.append(log_entry)
                    print(f"Logged accuracy: {log_entry}")
                    total_test_time += (time.time() - test_start_time)
                    model.train()
        
            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch + 1}/{num_epoch}, Average Loss: {avg_loss:.4f}")

        total_training_time = time.time() - start_time - total_test_time
        unlearn_acc = test_nlp(model, test_unlearn_dataloader)
        retain_acc = test_nlp(model, test_dataloader)
        log_entry = {
            "step": step,
            "training_time": preprocess_time + total_training_time,
            "unlearn_accuracy": unlearn_acc,
            "retain_accuracy": retain_acc
        }
        accuracy_log.append(log_entry)

    if task in CV_TASK:
        test_interval_steps = 200

        unlearn_acc = test_cv(model, test_unlearn_dataloader)
        retain_acc = test_cv(model, test_dataloader)
        log_entry = {
            "step": 0,
            "training_time": preprocess_time,
            "unlearn_accuracy": unlearn_acc,
            "retain_accuracy": retain_acc
        }
        accuracy_log.append(log_entry)

        for epoch in range(num_epoch):
            epoch_loss = 0
            for images, labels in dataloader:
                model.train()
                step += 1
                images, labels = images.to(device), labels.to(device)

                outputs = model(images).logits
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

                if step % test_interval_steps == 0:
                    test_start_time = time.time()
                    elapsed_training_time = time.time() - start_time - total_test_time
                    unlearn_acc = test_cv(model, test_unlearn_dataloader)
                    retain_acc = test_cv(model, test_dataloader)
                    log_entry = {
                        "step": step,
                        "training_time": preprocess_time + elapsed_training_time,
                        "unlearn_accuracy": unlearn_acc,
                        "retain_accuracy": retain_acc
                    }
                    accuracy_log.append(log_entry)
                    print(f"Logged accuracy: {log_entry}")
                    total_test_time += (time.time() - test_start_time)
                    model.train()

            avg_loss = epoch_loss / len(dataloader)
            print(f'Epoch [{epoch+1}/{num_epoch}], Loss: {avg_loss:.4f}')
        
        total_training_time = time.time() - start_time - total_test_time
        unlearn_acc = test_cv(model, test_unlearn_dataloader)
        retain_acc = test_cv(model, test_dataloader)
        log_entry = {
            "step": step,
            "training_time": preprocess_time + total_training_time,
            "unlearn_accuracy": unlearn_acc,
            "retain_accuracy": retain_acc
        }
        accuracy_log.append(log_entry)



    print("Result recorded")
$$;

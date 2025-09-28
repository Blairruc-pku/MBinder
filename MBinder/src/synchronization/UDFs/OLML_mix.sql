CREATE OR REPLACE PROCEDURE update_model_mix(model_name TEXT, test_table TEXT, args_name TEXT)
LANGUAGE plpython3u AS $$
    """
    Update model using Continue Learning / Machine Unlearning methods with mixed batches.
    Params:
        model_name: Name of a model. Suppose to appear in table 'finetuned_models'.
        args_name: Name of a set of args. Suppose to appear in table 'args'.
    """
    import ast
    import base64
    import copy
    import datetime
    import dill
    import json
    from functools import partial
    import re
    import numpy as np
    import pandas as pd
    import psycopg2
    from psycopg2.extensions import adapt
    import time
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as transforms
    from transformers import BertModel, BertTokenizer, AutoImageProcessor, AutoTokenizer
    from transformers import ViTForImageClassification, ViTFeatureExtractor

    import sys
    
    sys.path.insert(-1,'')

    from NLP_dataset import NLPDataset
    from CV_dataset import CVDataset
    from NLP_update import nlp_alternate_learn_unlearn, BERTClassifier, tab_alternate_learn_unlearn
    from CV_update import cv_alternate_learn_unlearn

    NLP_TASK = ['SentimentClsCH', 'SentimentClsEN', 'TextClass']
    CV_TASK = ['ImageClass2','ImageClass3','Digit', 'ImageClass']
    TAB_TASK = ['TabClass']

    seed = 42
    torch.manual_seed(seed)
    
    start_time = time.time()

    # Get model related info
    query = "SELECT * FROM FINETUNED_MODELS WHERE name='{}'".format(model_name)
    model_record = plpy.execute(query)[0]
    train_table_name = model_record['training_dataset']

    # Get args info
    query = "SELECT * FROM args_sync WHERE name='{}'".format(args_name)
    args = plpy.execute(query)[0]

    # Collect changed data since last update with WAL
    query = f"""
    SELECT lsn,
    CASE 
        WHEN data LIKE '%INSERT%' THEN 'INSERT' 
        WHEN data LIKE '%DELETE%' THEN 'DELETE' 
        ELSE 'UNKNOWN' 
    END AS change_type, 
    substring(data FROM 'text\\[text\\]:''((?:[^'']|'''')*)''') AS text_content,
    regexp_replace(
        substring(data FROM '(INSERT|DELETE): label\\[integer\\]:\\d+\\s*(.*)'),
        '\\[integer\\]:(\\d+)', '=\\1,', 'g'
    ) AS tabular_content, 
    encode(regexp_replace(substring(data FROM 'image\\[bytea\\]:''(.*?\\\\377\\\\331|.*?IEND\\\\256B`\\\\202)''')::text, '''''', '''', 'g')::bytea, 'base64') AS image_content, 
    substring(data FROM 'label\\[integer\\]:(\d+)')::INTEGER AS label_value 
    FROM pg_logical_slot_peek_changes('slot', NULL, NULL)
    WHERE data ~ '^table public\.{train_table_name}:'
    """
    data_record = list(plpy.execute(query))

    query = f"""
    WITH split_updates AS ( 
        SELECT 
            lsn, 'DELETE' AS change_type, 
            substring(data FROM 'old-key: text\\[text\\]:''((?:[^'']|'')*)''') AS text_content, 
            encode(regexp_replace(substring(data FROM 'image\\[bytea\\]:''(.*?\\\\377\\\\331|.*?IEND\\\\256B`\\\\202)''')::text, '''''', '''', 'g')::bytea, 'base64') AS image_content, 
            substring(data FROM 'old-key: .*?label\\[integer\\]:(\d+)')::INTEGER AS label_value 
        FROM pg_logical_slot_peek_changes('slot', NULL, NULL) 
        WHERE data LIKE '%UPDATE%' AND data ~ '^table public\.{train_table_name}:'
        UNION ALL 
        SELECT  
            lsn, 'INSERT' AS change_type, 
            substring(data FROM 'new-tuple: text\\[text\\]:''((?:[^'']|'')*)''') AS text_content, 
            encode(regexp_replace(substring(data FROM 'image\\[bytea\\]:''(.*?\\\\377\\\\331|.*?IEND\\\\256B`\\\\202)''')::text, '''''', '''', 'g')::bytea, 'base64') AS image_content, 
            substring(data FROM 'new-tuple: .*?label\\[integer\\]:(\d+)')::INTEGER AS label_value 
        FROM pg_logical_slot_peek_changes('slot', NULL, NULL) 
        WHERE data LIKE '%UPDATE%' AND data ~ '^table public\.{train_table_name}:'
    ) 
    SELECT lsn, change_type, text_content, image_content, label_value 
    FROM split_updates WHERE label_value IS NOT NULL
    """

    update_data_record = list(plpy.execute(query))
    data_record.extend(update_data_record)
    print("Change numbers: {}".format(len(data_record)))

    lsn_list = [item['lsn'] for item in data_record]
    if lsn_list:
        max_lsn = max(lsn_list)
        query = "SELECT pg_replication_slot_advance('slot', '{}');".format(max_lsn)
        plpy.execute(query)
    else:
        print("No changes detected.")

    """
    extract data
    """
    CL_max_label = 0
    data_insert = []
    label_insert = []
    data_delete = []
    label_delete = []
    insert_sample = args['learn_sample']
    delete_sample = args['unlearn_sample']
    for item in data_record:
        if item['change_type'] == "INSERT":
            if model_record['task'] in TAB_TASK:
                data_insert.append(item['tabular_content'])
            elif model_record['task'] in NLP_TASK:
                data_insert.append(item['text_content'])
            elif model_record['task'] in CV_TASK and item['image_content'] is not None:
                data_insert.append(base64.b64decode(item['image_content']))
            label_insert.append(int(item['label_value']))
        if item['change_type'] == "DELETE":
            if model_record['task'] in TAB_TASK:
                data_delete.append(item['tabular_content'])
            elif model_record['task'] in NLP_TASK:
                data_delete.append(item['text_content'])
            elif model_record['task'] in CV_TASK and item['image_content'] is not None:
                data_delete.append(base64.b64decode(item['image_content']))
            label_delete.append(int(item['label_value']))
    
    print(len(data_insert))
    print(len(label_insert))
    print(len(data_delete))
    print(len(label_delete))

    CL_max_label = max(label_insert)

    if insert_sample < 1:
        df = pd.DataFrame({"text":data_insert, "label":label_insert})
        sampled_df = df.groupby("label", group_keys=False).apply(lambda x: x.sample(frac=insert_sample, random_state=seed))
        data_insert, label_insert = sampled_df["text"].tolist(), sampled_df["label"].tolist()
    if delete_sample < 1:
        df = pd.DataFrame({"text":data_delete, "label":label_delete})
        sampled_df = df.groupby("label", group_keys=False).apply(lambda x: x.sample(frac=delete_sample, random_state=seed))
        data_delete, label_delete = sampled_df["text"].tolist(), sampled_df["label"].tolist()

    delete_label = tuple(np.unique(label_delete))
    if len(delete_label) == 1:
        delete_label = f"({delete_label[0]})"
    else:
        delete_label = str(delete_label)

    # get replay data for CL
    if model_record['task'] in TAB_TASK:
        query = "SELECT text, label FROM ( \
            SELECT ( \
                SELECT STRING_AGG(CONCAT(key, ': ', value), ',') \
                FROM jsonb_each_text(to_jsonb(frappe) - 'label') \
            ) AS text, label, ROW_NUMBER() OVER (PARTITION BY label ORDER BY RANDOM()) AS rn, \
            COUNT(*) OVER (PARTITION BY label) AS total_count \
            FROM {}) AS subquery WHERE rn <= total_count * {};".format(train_table_name, args['replay_sample'])
    if model_record['task'] in NLP_TASK:
        query = "WITH LabelCounts AS (SELECT label, COUNT(*) AS total_count FROM {} WHERE label NOT IN {} GROUP BY label),\
            RandomSample AS (SELECT a.text, a.label, RANDOM() AS rand_value, c.total_count FROM {} a JOIN LabelCounts c ON a.label = c.label),\
            RankedSample AS (SELECT *, ROW_NUMBER() OVER (PARTITION BY label ORDER BY rand_value) AS row_num FROM RandomSample),\
            FilteredSample AS (SELECT * FROM RankedSample WHERE row_num <= total_count * {})\
            SELECT * FROM FilteredSample".format(train_table_name, delete_label, train_table_name, args['replay_sample'])
    if model_record['task'] in CV_TASK:
        query = "WITH LabelCounts AS (SELECT label, COUNT(*) AS total_count FROM {} WHERE label NOT IN {} GROUP BY label),\
            RandomSample AS (SELECT encode(a.image, 'base64') AS image, a.label, RANDOM() AS rand_value, c.total_count FROM {} a JOIN LabelCounts c ON a.label = c.label),\
            RankedSample AS (SELECT *, ROW_NUMBER() OVER (PARTITION BY label ORDER BY rand_value) AS row_num FROM RandomSample),\
            FilteredSample AS (SELECT * FROM RankedSample WHERE row_num <= total_count * {})\
            SELECT * FROM FilteredSample".format(train_table_name, delete_label, train_table_name, args['replay_sample'])
    replayed_data = plpy.execute(query)
    for item in replayed_data:
        if model_record['task'] in NLP_TASK or model_record['task'] in TAB_TASK:
            data_insert.append(item['text'])
        elif model_record['task'] in CV_TASK:
            data_insert.append(base64.b64decode(item['image']))
        label_insert.append(item['label'])

    print(len(data_insert))
    print(len(label_insert))

    raw_test_data = []
    raw_test_unlearn_data = []
    if model_record['task'] in TAB_TASK:
        query2 = "SELECT ( \
            SELECT STRING_AGG(CONCAT(key, ': ', value), ',') \
            FROM jsonb_each_text(to_jsonb({}) - 'label') \
            ) AS text, label, userid FROM {} WHERE userid IN (SELECT DISTINCT userid FROM {}) ".format(test_table, test_table, train_table_name);
        raw_test_data = plpy.execute(query2)
        query3 = "SELECT ( \
            SELECT STRING_AGG(CONCAT(key, ': ', value), ',') \
            FROM jsonb_each_text(to_jsonb({}) - 'label') \
            ) AS text, label, userid FROM {} WHERE userid NOT IN (SELECT DISTINCT userid FROM {}) ".format(test_table, test_table, train_table_name);
        raw_test_unlearn_data = plpy.execute(query3)

    if model_record['task'] in NLP_TASK:
        query2 = "SELECT * FROM {} WHERE label IN (SELECT DISTINCT label FROM {}) ".format(test_table, train_table_name);
        raw_test_data = plpy.execute(query2)
        query3 = "SELECT * FROM {} WHERE label NOT IN (SELECT DISTINCT label FROM {}) ".format(test_table, train_table_name);
        raw_test_unlearn_data = plpy.execute(query3)
    
    if model_record['task'] in CV_TASK:
        query2 = "SELECT encode(image, 'base64') AS image, label FROM {} WHERE label IN (SELECT DISTINCT label FROM {})".format(test_table, train_table_name);
        raw_test_data = plpy.execute(query2)
        query3 = "SELECT encode(image, 'base64') AS image, label FROM {} WHERE label NOT IN (SELECT DISTINCT label FROM {})".format(test_table, train_table_name);
        raw_test_unlearn_data = plpy.execute(query3)

    test_data_list = []
    test_label_list = []
    test_unlearn_data_list = []
    test_unlearn_label_list = []

    if model_record['task'] in NLP_TASK or model_record['task'] in TAB_TASK:
        for item in raw_test_data:
            test_data_list.append(item['text'])
            test_label_list.append(item['label'])
        for item in raw_test_unlearn_data:
            test_unlearn_data_list.append(item['text'])
            test_unlearn_label_list.append(abs(item['label']))
    if model_record['task'] in CV_TASK:
        for item in raw_test_data:
            test_data_list.append(base64.b64decode(item['image']))
            test_label_list.append(item['label'])
        for item in raw_test_unlearn_data:
            test_unlearn_data_list.append(base64.b64decode(item['image']))
            test_unlearn_label_list.append(abs(item['label']))
    
    print(len(test_data_list))
    print(len(test_unlearn_data_list))

    """
    set model
    """
    model = None

    class BERTClassifier(nn.Module):
        def __init__(self, bert_model_path, num_labels):
            super(BERTClassifier, self).__init__()
            self.bert = BertModel.from_pretrained(bert_model_path)  
            self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
        def forward(self, input_ids, attention_mask):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            cls_output = outputs.pooler_output
            logits = self.classifier(cls_output)
            return logits

    if model_record['task'] in NLP_TASK:
        model = BERTClassifier("/home/taohonglin/model_repo/bert-base-uncased", num_labels = model_record["label_num"])
        model.load_state_dict(torch.load(model_record['path'] + "/model_state_dict.bin"))
    if model_record['task'] in CV_TASK or model_record['task'] in TAB_TASK:
        model = torch.load(model_record['path'] + "/pytorch_model.bin")

    """
    set dataloader
    """
    cl_dataloader = None
    ul_dataloader = None
    test_dataloader = None
    test_unlearn_dataloader = None
    if model_record['task'] in NLP_TASK or model_record['task'] in TAB_TASK:
        tokenizer = BertTokenizer.from_pretrained(model_record['path'])
        dataset = NLPDataset(texts=data_insert, labels=label_insert, tokenizer=tokenizer,max_len=256)
        cl_dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        dataset = NLPDataset(texts=data_delete, labels=label_delete, tokenizer=tokenizer,max_len=256)
        ul_dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        test_dataset = NLPDataset(texts=test_data_list, labels=test_label_list, tokenizer=tokenizer, max_len=256)
        test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        test_unlearn_dataset = NLPDataset(texts=test_unlearn_data_list, labels=test_unlearn_label_list, tokenizer=tokenizer, max_len=256)
        test_unlearn_dataloader = DataLoader(test_unlearn_dataset, batch_size=16, shuffle=False)
    if model_record['task'] in CV_TASK:
        processor = AutoImageProcessor.from_pretrained(model_record['path'])
        # DO NOT USE use_fast=True Which Causes FATAL ERROR

        dataset = CVDataset(images=data_insert, labels=label_insert, processor=processor)
        cl_dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        dataset = CVDataset(images=data_delete, labels=label_delete, processor=processor)
        ul_dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        test_dataset = CVDataset(images=test_data_list, labels=test_label_list, processor=processor)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        test_unlearn_dataset = CVDataset(images=test_unlearn_data_list, labels=test_unlearn_label_list, processor=processor)
        test_unlearn_dataloader = DataLoader(test_unlearn_dataset, batch_size=32, shuffle=False)

    # set trainable params
    num_layers = 0
    if hasattr(model, "bert"):
        num_layers = model.bert.encoder.config.num_hidden_layers
    if hasattr(model, "vit"):
        num_layers = model.vit.encoder.config.num_hidden_layers
    for name, param in model.named_parameters():
        param.requires_grad = False
        if "classifier" in name:
            param.requires_grad = True
    for i in range(num_layers // 2, num_layers):
        layer = None
        if hasattr(model, "bert"):
            layer = model.bert.encoder.layer[i] 
        elif hasattr(model, "vit"):
            layer = model.vit.encoder.layer[i] 
        for param in layer.output.dense.parameters():
            param.requires_grad = True
        for param in layer.intermediate.dense.parameters():
            param.requires_grad = True

    # set matched classifier
    new_classes_num = model.classifier.out_features
    if CL_max_label >= new_classes_num:
        print("New class detected, classifier will be changed.")
        new_classes_num = CL_max_label + 1
    
    preprocess_time = time.time() - start_time

    """
    Unlearning then Continue Learning
    """
    accuracy_log = []
    if model_record['task'] in TAB_TASK:
        model, _ = tab_alternate_learn_unlearn(
            model=model,
            ul_dataloader=ul_dataloader,
            cl_dataloader=cl_dataloader,
            test_dataloader=test_dataloader,
            test_unlearn_dataloader=test_unlearn_dataloader,
            new_class_num=new_classes_num,
            unlearn_epochs=int(args['unlearn_epoch']),
            learn_epochs=int(args['learn_epoch']),
            unlearn_lr=float(args['unlearn_rate']),
            learn_lr=float(args['learn_rate']),
            weight_decay=float(args['learn_weight_decay']),
            lambda_=float(args['lambda'])
        )
        for item in _:
            item['training_time'] += preprocess_time
            accuracy_log.append(item)
    if model_record['task'] in NLP_TASK:
        model, _ = nlp_alternate_learn_unlearn(
            model=model,
            ul_dataloader=ul_dataloader,
            cl_dataloader=cl_dataloader,
            test_dataloader=test_dataloader,
            test_unlearn_dataloader=test_unlearn_dataloader,
            new_class_num=new_classes_num,
            unlearn_epochs=int(args['unlearn_epoch']),
            learn_epochs=int(args['learn_epoch']),
            unlearn_lr=float(args['unlearn_rate']),
            learn_lr=float(args['learn_rate']),
            weight_decay=float(args['learn_weight_decay']),
            lambda_=float(args['lambda'])
        )
        for item in _:
            item['training_time'] += preprocess_time
            accuracy_log.append(item)
    if model_record['task'] in CV_TASK:
        model, _ = cv_alternate_learn_unlearn(
            model=model,
            ul_dataloader=ul_dataloader,
            cl_dataloader=cl_dataloader,
            test_dataloader=test_dataloader,
            test_unlearn_dataloader=test_unlearn_dataloader,
            new_class_num=new_classes_num,
            unlearn_epochs=int(args['unlearn_epoch']),
            learn_epochs=int(args['learn_epoch']),
            unlearn_lr=float(args['unlearn_rate']),
            learn_lr=float(args['learn_rate']),
            weight_decay=float(args['learn_weight_decay']),
            lambda_=float(args['lambda'])
        )
        for item in _:
            item['training_time'] += preprocess_time
            accuracy_log.append(item) 

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


    print("Result recorded")
    

$$;
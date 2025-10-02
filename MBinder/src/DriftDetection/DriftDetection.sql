CREATE OR REPLACE FUNCTION DriftDetection(model_name TEXT)
RETURNS BOOLEAN AS $$
    """
    Detect whether the train table encountered Drifting
    From 2 ways:
    label distribution
    feature distribution
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
    from scipy.stats import entropy
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
    from transformers import BertModel, BertTokenizer, AutoImageProcessor, AutoTokenizer
    from transformers import ViTForImageClassification, ViTFeatureExtractor, PreTrainedModel

    import sys
    
    sys.path.insert(-1,'/home/thl/MBinder/OLML')

    from OLML_postgres.cron.NLP_dataset import NLPDataset
    from OLML_postgres.cron.CV_dataset import CVDataset
    from OLML_postgres.cron.NLP_update import nlp_alternate_learn_unlearn, BERTClassifier, tab_alternate_learn_unlearn
    from OLML_postgres.cron.CV_update import cv_alternate_learn_unlearn

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

    # Detect label drift
    query = "SELECT DISTINCE label FROM {}".format(table_name)
    labels = sort(list(plpy.execute(query)))

    query = "SELECT DISTINCE label FROM _{}".format(table_name)
    buffer_labels = sort(list(plpy.execute(query)))

    
    if len(labels) != len(buffer_labels):
        return True
    
    for label1, label2 in zip(labels, buffer_labels):
        if label1 != label2
            return True
    
    # Detect feature drift

    # Access inserted data and deleted data
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
    
    if data_record = []:
        return False
    
    data_insert = []
    label_insert = []
    data_delete = []
    label_delete = []

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
    
    buffer_data = []
    buffer_data_label = []
    if model_record['task'] in NLP_TASK:
        query = "SELECT * FROM {}".format(train_table_name);
        data = plpy.execute(query)
        for item in data:
            buffer_data.append(item['text'])
            buffer_data_label.append(item['label'])
    if model_record['task'] in CV_TASK:
        query = "SELECT encode(image, 'base64')as image, label FROM {}".format(train_table_name);
        data = plpy.execute(query)
        for item in data:
            buffer_data.append(item['text'])
            buffer_data_label.append(item['label'])
    
    # Access model and preprocess
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
    elif model_record['task'] in CV_TASK:
        model = torch.load(model_record['path'] + "/pytorch_model.bin")

    if model_record['task'] in NLP_TASK:
        tokenizer = BertTokenizer.from_pretrained(model_record['path'])
    elif model_record['task'] in CV_TASK:
        transform = transforms.Compose([
            transforms.Resize((model.config.image_size, model.config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    
    # get buffer embedding
    label_probabilities = {}

    for text, label in zip(buffer_data, buffer_data_label):
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy().squeeze()
        label_probabilities[label].append(probabilities)

    # get buffer info
    for label, probs_list in label_probabilities.items():
        probs_array = np.vstack(probs_list)
        mean_prob_vector = np.mean(probs_array, axis=0)
        
        max_kl = 0
        for prob_vec in probs_array:
            p = np.clip(prob_vec, 1e-10, 1)
            q = np.clip(mean_prob_vector, 1e-10, 1)
            
            kl = entropy(p, q) if np.any(p > 0) and np.any(q > 0) else float('inf')
            max_kl = max(max_kl, kl)
        
        class_stats[label] = {
            'mean_prob_vector': mean_prob_vector,
            'max_kl': max_kl
        }
    
    # check inserted data
    drift_samples = 0

    for text in data_insert:
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            new_probabilities = torch.softmax(logits, dim=-1).cpu().numpy().squeeze()
        
        is_drift = True
        for label, stats in class_stats.items():
            p = np.clip(new_probabilities, 1e-10, 1)
            q = np.clip(stats['mean_prob_vector'], 1e-10, 1)
            
            kl = entropy(p, q)
            if kl <= stats['max_kl']:
                is_drift = False
                break
        
        if is_drift:
            drift_samples += 1

    drift_ratio = drift_samples / len(insert_data)

    if drift_ratio > 0.1:
        print("Feature drifting detected")
        return True

    # checking deleted data need current buffer instead of the old
    buffer_data = []
    buffer_data_label = []
    if model_record['task'] in NLP_TASK:
        query = f"""
            SELECT *
            FROM (
                SELECT text, label,
                    ROW_NUMBER() OVER (PARTITION BY label ORDER BY random()) AS rn,
                    COUNT(*) OVER (PARTITION BY label) AS total_count
                FROM {train_table_name}
            ) ranked
            WHERE rn <= CASE WHEN total_count*0.1 < 100 THEN total_count*0.1 ELSE 100 END
            )
            """
        data = plpy.execute(query)
        for item in data:
            buffer_data.append(item['text'])
            buffer_data_label.append(item['label'])
    if model_record['task'] in CV_TASK:
        query = f"""
        SELECT *
        FROM (
            SELECT encode(image, 'base64')as image, label,
                ROW_NUMBER() OVER (PARTITION BY label ORDER BY random()) AS rn,
                COUNT(*) OVER (PARTITION BY label) AS total_count
            FROM {train_table_name}
        ) ranked
        WHERE rn <= CASE WHEN total_count*0.1 < 100 THEN total_count*0.1 ELSE 100 END
        )
        """
        data = plpy.execute(query)
        for item in data:
            buffer_data.append(item['text'])
            buffer_data_label.append(item['label'])

    # get buffer embedding
    label_probabilities = {}

    for text, label in zip(buffer_data, buffer_data_label):
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy().squeeze()
        label_probabilities[label].append(probabilities)

    # get buffer info
    for label, probs_list in label_probabilities.items():
        probs_array = np.vstack(probs_list)
        mean_prob_vector = np.mean(probs_array, axis=0)
        
        max_kl = 0
        for prob_vec in probs_array:
            p = np.clip(prob_vec, 1e-10, 1)
            q = np.clip(mean_prob_vector, 1e-10, 1)
            
            kl = entropy(p, q) if np.any(p > 0) and np.any(q > 0) else float('inf')
            max_kl = max(max_kl, kl)
        
        class_stats[label] = {
            'mean_prob_vector': mean_prob_vector,
            'max_kl': max_kl
        }

    # check deleted data
    drift_samples = 0

    for text in data_delete:
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            new_probabilities = torch.softmax(logits, dim=-1).cpu().numpy().squeeze()
        
        is_drift = True
        for label, stats in class_stats.items():
            p = np.clip(new_probabilities, 1e-10, 1)
            q = np.clip(stats['mean_prob_vector'], 1e-10, 1)
            
            kl = entropy(p, q)
            if kl <= stats['max_kl']:
                is_drift = False
                break
        
        if is_drift:
            drift_samples += 1

    drift_ratio = drift_samples / len(insert_data)

    if drift_ratio > 0.1:
        print("Feature drifting detected")
        return True

    print("No Drifting")
    return False
$$ LANGUAGE plpython3u;
CREATE OR REPLACE PROCEDURE TRAILS(target_table TEXT, model_table TEXT)
LANGUAGE plpython3u AS $$
    """
    Select ideal model for target table fine-tuning.
    This is used for comparison.
    TRAILS selects model with 2 phases.
    Phase 1: filter model with trainability and express ability
    Phase 2  filter model with SH.
    """
    import time
    import json
    import os
    import datetime
    import numpy as np
    import pandas as pd
    import sys
    import torch

    
    from op.model import DNNModel

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    start_time = time.time()

    # get related tables
    plpy.execute("SELECT setseed(0.42)")
    query = "SELECT * FROM PRE_MODELS WHERE name IN (SELECT name FROM {})".format(model_table)
    model_info = plpy.execute(query)

    query = "SELECT text, label FROM ( \
    SELECT text, label, ROW_NUMBER() OVER (PARTITION BY label ORDER BY RANDOM()) AS rn \
    FROM {}) AS subquery WHERE rn <= 1000;".format(target_table)
    TARGET_TABLE = plpy.execute(query)
    print("Tables loaded.")

    query = "SELECT COUNT(DISTINCT label) FROM {}".format(target_table)
    label_num = plpy.execute(query)[0]['count']

    print("----------TRAILS PHASE 1----------")

    data_by_label = {label: [] for label in range(label_num)} 
    for row in TARGET_TABLE:
        data_by_label[row["label"]].append(row["text"])

    """
    Calculate Each Model's JacFlow
    To make a fair comparison, We sample the same number of data in each label
    and use average proxy_score as final score for filtering.
    Left the same number of models for Phase 2.
    """
    alpha = 0.8
    model_score = {}
    for item in model_info:
        model_record = [item['name'], target_table, item['path']]
        model = DNNModel(model_record, label_num)
        model.model.config.output_hidden_states = True

        losses = []
        log_k_h_values = []

        for idx in range(1000):
            selected_texts = [data_by_label[label][idx] for label in data_by_label]
            selected_labels = list(data_by_label.keys())

            inputs = model.tokenizer(selected_texts, padding=True, truncation=True, return_tensors="pt").to(device)
            labels_tensor = torch.tensor(selected_labels).to(device)

            with torch.no_grad():
                outputs = model.model(**inputs, labels=labels_tensor)
                loss = outputs.loss.item()
                logits = outputs.logits

                hidden_states = outputs.hidden_states[-1] if hasattr(outputs, "hidden_states") else None
                cls_activations = hidden_states[:, 0, :]  # [CLS] 激活值
                relu_activations = torch.relu(cls_activations)
                binary_pattern = (relu_activations > 0).int()

                num_samples = len(selected_texts)
                K_H = torch.zeros((num_samples, num_samples))
                for i in range(num_samples):
                    for j in range(i, num_samples):
                        hamming_dist = (binary_pattern[i] != binary_pattern[j]).sum().item()
                        K_H[i, j] = K_H[j, i] = binary_pattern.shape[1] - hamming_dist  # N_a = hidden_dim

                det_k_h = torch.linalg.det(K_H)
                log_K_H = torch.log2(det_k_h).item()

            losses.append(loss)
            log_k_h_values.append(log_K_H)

        model_params = torch.cat([p.contiguous().view(-1) for p in model.model.parameters()])

        loss_tensor = torch.tensor(loss).to(device)
        hadamard_product = loss_tensor * model_params

        average_hadamard = torch.mean(hadamard_product).item()

        average_log_k_h = np.mean(log_k_h_values)
        proxy_score = alpha * average_hadamard + (1 - alpha) * average_log_k_h

        model_score[item['name']] = proxy_score

        del model, model_params, hadamard_product, losses, log_k_h_values

    model_score = sorted(model_score.items(), key = lambda kv:(kv[1], kv[0]))
    print("model with score:")
    print(model_score)
    deleted_model = [item[0] for item in model_score[:len(model_score)-10]]
    model_info = [item for item in model_info if item['name'] not in deleted_model]

    del data_by_label
    print(model_info)
    print("----------TRAILS PHASE 2----------")
    """
    Select Model with SH.
    To make a fair comparison, set rate eta = 0.5
    """
    temp = {'text':[], 'label':[]}
    for item in TARGET_TABLE:
        temp['text'].append(item['text'])
        temp['label'].append(item['label'])

    label_groups = {}

    for text, label in zip(temp['text'], temp['label']):
        if label not in label_groups:
            label_groups[label] = []
        label_groups[label].append(text)

    train_data = {'text':[], 'label':[]}
    val_data = {'text':[], 'label':[]}

    for label, texts in label_groups.items():
        train_data['text'].extend(texts[:800]) 
        train_data['label'].extend([label] * 800)
        val_data['text'].extend(texts[800:1000])
        val_data['label'].extend([label] * 200)

    del temp,label_groups

    epoch = 0
    args = {'learn_rate':1e-5, 'learn_epoch':1}
    while len(model_info) > 1:
        print("----------epoch {}----------".format(epoch))
        epoch_acc = {}
        for i, item in enumerate(model_info):
            model_record = [item['name'], target_table, item['path']]
            model = DNNModel(model_record, label_num)
            epoch_acc[item['name']] = model.finetune(train_data, val_data, args)
            
            output_path = item['path'] + '/epoch{}'.format(epoch)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            model.save_model(output_path)
            model_info[i]['path'] = output_path
            del model
        
        epoch_acc = sorted(epoch_acc.items(), key = lambda kv:(kv[1], kv[0]))
        # delete last half
        deleted_model = [item[0] for item in epoch_acc[:len(epoch_acc)//2]]
        model_info = [item for item in model_info if item['name'] not in deleted_model]
        epoch += 1
    
    print("Model {} selected.".format(model_info[0]['name']))

    end_time = time.time()
    log = []
    item = {
        "time": end_time - start_time,
        "info": model_info[0]['name']
    }
    log.append(item)
$$;
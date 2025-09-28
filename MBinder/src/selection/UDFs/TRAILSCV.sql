CREATE OR REPLACE PROCEDURE TRAILSCV(target_table TEXT, model_table TEXT, sample_num INTEGER)
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
    import base64
    import io
    import os
    import datetime
    import numpy as np
    import pandas as pd
    from PIL import Image
    import sys
    import torch

    
    from op.model import ImageClassModel

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    start_time = time.time()

    # get related tables
    plpy.execute("SELECT setseed(0.42)")
    query = "SELECT * FROM PRE_MODELS WHERE name IN (SELECT name FROM {})".format(model_table)
    model_info = plpy.execute(query)

    query = "SELECT image, label FROM ( \
    SELECT encode(image, 'base64') as image, label, ROW_NUMBER() OVER (PARTITION BY label ORDER BY RANDOM()) AS rn \
    FROM {}) AS subquery WHERE rn <= {};".format(target_table, str(sample_num))
    TARGET_TABLE = plpy.execute(query)
    for data in TARGET_TABLE:
        decoded_image = base64.b64decode(data['image'])
        data['image'] = Image.open(io.BytesIO(decoded_image)).convert("RGB")
    print("Tables loaded.")

    query = "SELECT COUNT(DISTINCT label) FROM {}".format(target_table)
    label_num = plpy.execute(query)[0]['count']

    print("----------TRAILS PHASE 1----------")

    data_by_label = {label: [] for label in range(label_num)} 
    for row in TARGET_TABLE:
        data_by_label[row["label"]].append(row["image"])

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
        model = ImageClassModel(model_record, label_num)
        model.model.config.output_hidden_states = True

        losses = []
        log_k_h_values = []

        for idx in range(sample_num):
            selected_image = [data_by_label[label][idx] for label in data_by_label]
            selected_labels = list(data_by_label.keys())

            inputs = model.image_processor(selected_image, return_tensors="pt").to(device)
            labels_tensor = torch.tensor(selected_labels).to(device)

            with torch.no_grad():
                outputs = model.model(**inputs, labels=labels_tensor)
                loss = outputs.loss.item()
                logits = outputs.logits

                hidden_states = outputs.hidden_states[-1] if hasattr(outputs, "hidden_states") else None
                cls_activations = hidden_states[:, 0, :]  # [CLS] 激活值
                relu_activations = torch.relu(cls_activations)
                binary_pattern = (relu_activations > 0).int()

                num_samples = len(selected_image)
                K_H = torch.zeros((num_samples, num_samples))
                for i in range(num_samples):
                    for j in range(i, num_samples):
                        hamming_dist = (binary_pattern[i] != binary_pattern[j]).sum().item()
                        K_H[i, j] = K_H[j, i] = binary_pattern.shape[1] - hamming_dist  # N_a = hidden_dim

                sign, log_det_k_h = torch.linalg.slogdet(K_H)
                log_K_H = log_det_k_h.item() / torch.log(torch.tensor(2.0))
                if log_K_H < 0 :
                    log_K_H = 1e-9

            losses.append(loss)
            log_k_h_values.append(log_K_H)

        model_params = torch.cat([p.contiguous().view(-1) for p in model.model.parameters()])
        if torch.isnan(model_params).any() or torch.isinf(model_params).any():
            model_params = torch.nan_to_num(model_params, nan=0.0, posinf=1e10, neginf=-1e10)

        loss_tensor = torch.tensor(loss).to(device)
        hadamard_product = loss_tensor * model_params

        average_hadamard = torch.mean(hadamard_product).item()

        average_log_k_h = np.mean(log_k_h_values)
        print(average_hadamard)
        print(average_log_k_h)
        proxy_score = alpha * average_hadamard + (1 - alpha) * average_log_k_h

        model_score[item['name']] = proxy_score

        del model, losses, log_k_h_values

    model_score = sorted(model_score.items(), key = lambda kv:(kv[1], kv[0]))
    print("model with score")
    print(model_score)
    deleted_model = [item[0] for item in model_score[:len(model_score)-10]]
    model_info = [item for item in model_info if item['name'] not in deleted_model]

    del data_by_label

    print("----------TRAILS PHASE 2----------")
    """
    Select Model with SH.
    To make a fair comparison, set rate eta = 0.5
    """
    train_data = []
    val_data = []
    count = [0] * label_num
    for item in TARGET_TABLE:
        if count[int(item['label'])] < int(sample_num*0.8):
            train_data.append({'img':item['image'], 'label':item['label']})
            count[int(item['label'])] += 1
        else:
            val_data.append({'img':item['image'], 'label':item['label']})

    epoch = 0
    args = {'learn_rate':1e-5, 'learn_epoch':1}
    while len(model_info) > 1:
        print("----------epoch {}----------".format(epoch))
        epoch_acc = {}
        for i, item in enumerate(model_info):
            model_record = [item['name'], target_table, item['path']]
            model = ImageClassModel(model_record, label_num)
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
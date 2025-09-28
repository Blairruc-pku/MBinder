CREATE OR REPLACE PROCEDURE SuccessiveHalving(target_table TEXT, model_table TEXT)
LANGUAGE plpython3u AS $$
    """
    Select ideal model for target table fine-tuning.
    This is used for comparison as a baseline method.
    SH filter half of models at each epoch.
    """
    import time
    import json
    import os
    import datetime
    import pandas as pd
    import sys

    from re_coarse_recall_v2 import CoarseRecall
    from re_fine_selection import FineSelection
    from op.model import DNNModel

    start_time = time.time()

    # get related tables
    plpy.execute("SELECT setseed(0.42)")
    query = "SELECT * FROM PRE_MODELS WHERE name IN (SELECT name FROM {})".format(model_table)
    model_info = plpy.execute(query)
    print(len(model_info))

    query = "SELECT text, label FROM ( \
    SELECT text, label, ROW_NUMBER() OVER (PARTITION BY label ORDER BY RANDOM()) AS rn \
    FROM {}) AS subquery WHERE rn <= 1000;".format(target_table)
    TARGET_TABLE = plpy.execute(query)
    print("Tables loaded.")

    query = "SELECT COUNT(DISTINCT label) FROM {}".format(target_table)
    label_num = plpy.execute(query)

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
            model = DNNModel(model_record, label_num[0]['count'])
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
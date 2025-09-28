CREATE OR REPLACE PROCEDURE SuccessiveHalvingCV(target_table TEXT, model_table TEXT, sample_num INTEGER)
LANGUAGE plpython3u AS $$
    """
    Select ideal model for target table fine-tuning.
    This is used for comparison as a baseline method.
    SH filter half of models at each epoch.
    """
    import base64
    import os
    import datetime
    import io
    import time
    import json
    import pandas as pd
    from PIL import Image
    import sys

    from op.model import ImageClassModel

    start_time = time.time()

    # get related tables
    plpy.execute("SELECT setseed(0.42)")
    query = "SELECT * FROM PRE_MODELS WHERE name IN (SELECT name FROM {})".format(model_table)
    model_info = plpy.execute(query)
    print(len(model_info))

    query = "SELECT image, label FROM ( \
    SELECT encode(image, 'base64') as image, label, ROW_NUMBER() OVER (PARTITION BY label ORDER BY RANDOM()) AS rn \
    FROM {}) AS subquery WHERE rn <= {};".format(target_table, sample_num)
    TARGET_TABLE = plpy.execute(query)
    for data in TARGET_TABLE:
        decoded_image = base64.b64decode(data['image'])
        data['image'] = Image.open(io.BytesIO(decoded_image)).convert("RGB")
    print("Tables loaded.")

    query = "SELECT COUNT(DISTINCT label) FROM {}".format(target_table)
    label_num = plpy.execute(query)

    train_data = []
    val_data = []
    count = [0] * label_num[0]['count']
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
            model = ImageClassModel(model_record, label_num[0]['count'])
            epoch_acc[item['name']] = model.finetune(train_data, val_data, args)
            
            output_path = item['path'] + '/epoch{}'.format(epoch)
            if not os.path.exists(output_path):
                os.makedirs(output_path, exist_ok=True)
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
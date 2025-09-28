CREATE OR REPLACE PROCEDURE ModelSelectionCV(target_table TEXT, sample_num INTEGER)
LANGUAGE plpython3u AS $$
    """
    Select ideal model for target table fine-tuning.
    Reference table: target_table, PERF_MATRIX, CONV_TREND
    ModelSelection can only apply to classifier because of LEEP.
    Target_table should have two columns: image and label.
    """
    import base64
    import gc
    import io
    import logging
    import time
    import json
    import pandas as pd
    from PIL import Image
    import sys

    from re_coarse_recall_v2 import CoarseRecall
    from re_fine_selection import FineSelection

    logging.getLogger("natten.functional").setLevel(logging.ERROR)

    start_time = time.time()

    # get related tables
    plpy.execute("SELECT setseed(0.42)")
    query = "SELECT * FROM PERF_MATRIX_CV"
    PERF_MATRIX = plpy.execute(query)
    query = "SELECT * FROM CONV_TREND_CV"
    CONV_MATRIX = plpy.execute(query)
    # todo: CONV_TREND_FINAL_CV to DB
    query = "SELECT * FROM CONV_TREND_FINAL_CV"
    CONV_FINAL = plpy.execute(query)
    query = "SELECT image, label FROM ( \
    SELECT encode(image, 'base64') as image, label, ROW_NUMBER() OVER (PARTITION BY label ORDER BY RANDOM()) AS rn \
    FROM {}) AS subquery WHERE rn <= {};".format(target_table, str(sample_num))
    TARGET_TABLE = plpy.execute(query)
    for data in TARGET_TABLE:
        decoded_image = base64.b64decode(data['image'])
        data['image'] = Image.open(io.BytesIO(decoded_image)).convert("RGB")
    
    query = "SELECT COUNT(DISTINCT label) FROM {}".format(target_table)
    label_num = plpy.execute(query)
    print("Tables loaded.")
    
    """
    First phase (coarse recall) uses PERF_MATRIX and TARGET_TABLE
    """
    query = "SELECT * FROM PRE_MODELS"
    model_record = plpy.execute(query)
    coarse_recall = CoarseRecall(target_table, TARGET_TABLE, PERF_MATRIX, model_record, 'CV')
    recall_result = coarse_recall.recall()
    print(recall_result)
    print("First phase ended.")
    print("-" * 20)
    del coarse_recall

    """
    Second phase (fine selection)
    The result shapes like {'target_table': {epoch_num0 : {info}, {epoch_num1 : {info}}}}
    The last epoch remains only one model.
    """
    
    train_data = []
    val_data = []
    count = [0] * label_num[0]['count']
    for item in TARGET_TABLE:
        if count[int(item['label'])] < int(sample_num*0.8):
            train_data.append({'img':item['image'], 'label':item['label']})
            count[int(item['label'])] += 1
        else:
            val_data.append({'img':item['image'], 'label':item['label']})

    print(train_data[0])

    del TARGET_TABLE
    gc.collect()

    fine_selection = FineSelection(CONV_FINAL, CONV_MATRIX, model_record, train_data, val_data, label_num[0]['count'], 'CV')
    """
    result = fine_selection.filter_models(recall_result, 'mnli', num_models=10, threshold=0.1)
    """
    result, _ = fine_selection.filter_models(recall_result, target_table, num_models=10, threshold=-0.05)
    print(result)
    end_time = time.time()
    log = []
    item = {
        "time": end_time - start_time,
        "info": result
    }
    log.append(item)

$$;

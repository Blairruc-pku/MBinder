CREATE OR REPLACE PROCEDURE ModelSelectionTAB(target_table TEXT)
LANGUAGE plpython3u AS $$
    """
    The only difference between TAB and text is the way get table content
    """
    import time
    import json
    import pandas as pd
    import sys

    from re_coarse_recall_v2 import CoarseRecall
    from re_fine_selection import FineSelection

    start_time = time.time()

    # get related tables
    plpy.execute("SELECT setseed(0.42)")
    query = "SELECT * FROM PERF_MATRIX_TAB"
    PERF_MATRIX = plpy.execute(query)
    query = "SELECT * FROM CONV_TREND_TAB"
    CONV_MATRIX = plpy.execute(query)
    CONV_FINAL = PERF_MATRIX
    query = "SELECT text, label FROM ( \
    SELECT ( \
        SELECT STRING_AGG(CONCAT(key, ': ', value), ',') \
        FROM jsonb_each_text(to_jsonb(frappe) - 'label') \
    ) AS text, label, ROW_NUMBER() OVER (PARTITION BY label ORDER BY RANDOM()) AS rn \
    FROM {}) AS subquery WHERE rn <= 1000;".format(target_table)
    TARGET_TABLE = plpy.execute(query)
    query = "SELECT COUNT(DISTINCT label) FROM {}".format(target_table)
    label_num = plpy.execute(query)
    print("Tables loaded.")
    
    """
    First phase (coarse recall) uses PERF_MATRIX and TARGET_TABLE
    """
    query = "SELECT * FROM PRE_MODELS_TAB"
    model_record = plpy.execute(query)
    coarse_recall = CoarseRecall(target_table, TARGET_TABLE, PERF_MATRIX, model_record)
    recall_result = coarse_recall.recall()
    print(recall_result)
    print("First phase ended.")
    print("-" * 20)

    """
    Second phase (fine selection)
    The result shapes like {'target_table': {epoch_num0 : {info}, {epoch_num1 : {info}}}}
    The last epoch remains only one model.
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

    fine_selection = FineSelection(CONV_FINAL, CONV_MATRIX, model_record, train_data, val_data, label_num[0]['count'])

    result, _ = fine_selection.filter_models(recall_result, target_table, num_models=5, threshold=-0.05)
    print(result)
    end_time = time.time()
    log = []
    item = {
        "time": end_time - start_time,
        "info": result
    }
    log.append(item)

$$;

import pandas as pd
import json
from sklearn import metrics
import warnings
import copy
warnings.filterwarnings("ignore")
import os
import codecs
from sklearn.cluster import KMeans
import numpy as np
from op.model import DNNModel, ImageClassModel
import math

FILE_PATH = os.path.dirname(__file__)

class FineSelection:
    def __init__(self, PERF_MATRIX, CONV_TREND, model_record, train_data, val_data, label_num, task_type='NLP'):
        """
        df_dic refers to the final test acc.
        df_dic_val refers to the conv trend.
        CONV_TREND is a plpy select query result.
        It shapes like [{},{},...{}]
        Each dict(row) is a model where the value can be a list.
        The list refers to the acc of after each epoch.
        """
        self.task_type = task_type
        self.model_record = model_record
        self.train_data = train_data
        
        self.val_data = val_data
        self.label_num = label_num

        self.df_dic = {}
        self.df_dic_val = {}
        self.all_model_name = []
        self.dataset_list = []

        for model_perf in PERF_MATRIX:
            self.all_model_name.append(model_perf['model_name'])
            test_acc = []
            for key, value in model_perf.items():
                if key != 'model_name':
                    test_acc.append(value)
            self.df_dic[model_perf['model_name']] = test_acc

        for model_trend in CONV_TREND:
            model_name = model_trend['model_name']
            model_trend_copy = copy.deepcopy(model_trend)
            del model_trend_copy['model_name']
            self.dataset_list = list(model_trend_copy.keys())
            self.df_dic_val[model_name] = model_trend_copy


    def create_cluster_model_new(self, target_model, target_dataset, target_epoch, K=4):
        x_train = []  
        y_train = []  

        for i, dataset in enumerate(self.df_dic_val[target_model].keys()):
            if dataset == target_dataset:
                continue
            x_train.append(self.df_dic_val[target_model][dataset][target_epoch])
            y_train.append(self.df_dic[target_model][i])
        x_train = np.array(x_train).reshape(-1, 1)

        kmeans = KMeans(n_clusters=K)
        labels = kmeans.fit_predict(x_train)

        cluster_means = {}
        for i in range(4):
            cluster_means[i] = np.mean([y for y, label in zip(y_train, labels) if label == i])

        return kmeans, cluster_means

    def filter_models(self, coarse_recall_result, target_dataset_name, num_models=10, threshold=0.1):
        results = {} 
        target_dataset_list = [target_dataset_name]

        print(self.dataset_list)
        for dataset_index, target_dataset in enumerate(target_dataset_list):
            top_models = coarse_recall_result[:num_models]
            results[target_dataset] = {}
            real_model = {}
            
            train_flag = False

            if target_dataset not in self.dataset_list:
                """
                This means that current dataset has not been used for training
                Load them for training
                """
                train_flag = True
                for model_name in top_models:
                    print(model_name)
                    self.df_dic_val[model_name][target_dataset] = []

                    record = []
                    for temp in self.model_record:
                        if temp['name'] == model_name:
                            record = [temp['name'], 'training_dataset', temp['path'], 'task']
                            break
                    model = None
                    if self.task_type == 'NLP':
                        model = DNNModel(record, self.label_num)
                    if self.task_type == 'CV':
                        model = ImageClassModel(record, self.label_num)
                    model.unload_model()
                    real_model[model_name] = model
            print(len(real_model))

            for epoch in range(5):  
                print("----------epoch{}----------".format(str(epoch)))
                if train_flag:
                    """
                    For each left model, train one epoch 
                    """

                    args = {'learn_epoch':1}
                    for model_name in top_models:
                        print("Finetuning {}".format(model_name))
                        acc = real_model[model_name].finetune(self.train_data, self.val_data, args)
                        real_model[model_name].unload_model()
                        self.df_dic_val[model_name][target_dataset].append(acc)

                cur_num_models = len(top_models)

                predictions = {}
                for target_model in top_models:
                    cluster_model, cluster_means = self.create_cluster_model_new(target_model, target_dataset, epoch, 3)

                    val_accuracy = self.df_dic_val[target_model][target_dataset][epoch]
                    val_accuracy = np.array([val_accuracy]).reshape(-1, 1)

                    prediction = cluster_model.predict(val_accuracy)
                    predictions[target_model] = cluster_means[prediction[0]]

                max_per = max(self.df_dic_val[x][target_dataset][epoch] for x in predictions)
                for i, target_model in enumerate(
                        sorted(predictions, key=lambda x: self.df_dic_val[x][target_dataset][epoch])):
                    if self.df_dic_val[target_model][target_dataset][epoch] == max_per:
                        continue
                    #   if predictions[target_model] < max(predictions[model] for model in top_models if df_dic_val[model][target_dataset][epoch] > df_dic_val[target_model][target_dataset][epoch])-threshold:
                    #       top_models.remove(target_model)
                    max_res = max(predictions[model] for model in top_models if
                                  self.df_dic_val[model][target_dataset][epoch] >
                                  self.df_dic_val[target_model][target_dataset][
                                      epoch])

                    if (max_res - predictions[target_model]) / max_res >= threshold:
                        top_models.remove(target_model)

                while len(top_models) > cur_num_models // 2:
                    worst_model = min(top_models, key=lambda model: self.df_dic_val[model][target_dataset][epoch])
                    top_models.remove(worst_model)
                    real_model.pop(worst_model)

                results[target_dataset][epoch] = {
                    "left_models": list(top_models),
                    "left_num_models": len(top_models),
                    "best_test_performance": max(self.df_dic_val[model_name][target_dataset][epoch] for model_name in top_models)
                }
                print("current result:")
                print(results[target_dataset][epoch])

                if len(top_models) == 1:
                    print("Model: %s selected with performance: %s" %(list(top_models)[0], self.df_dic_val[top_models[0]][target_dataset][-1]))
                    break
        return results, real_model[list(top_models)[0]]


if __name__ == "__main__":
    fineSelection = FineSelection()
    threshold = 0.1
    coarse_recall_result = ['ishan--bert-base-uncased-mnli', 'Jeevesh8--feather_berts_46', 'emrecan--bert-base-multilingual-cased-snli_tr',
     'gchhablani--bert-base-cased-finetuned-rte', 'XSY--albert-base-v2-imdb-calssification', 'Jeevesh8--bert_ft_qqp-9',
     'Jeevesh8--bert_ft_qqp-40', 'connectivity--bert_ft_qqp-1', 'gchhablani--bert-base-cased-finetuned-wnli',
     'Jeevesh8--bert_ft_qqp-68', 'connectivity--bert_ft_qqp-96', 'classla--bcms-bertic-parlasent-bcs-ter',
     'Jeevesh8--init_bert_ft_qqp-33', 'connectivity--bert_ft_qqp-17', 'Jeevesh8--init_bert_ft_qqp-24']
    filter_results_0 = fineSelection.filter_models(coarse_recall_result, num_models=10, threshold=threshold)

    print(filter_results_0)
    """
    for k in filter_results_0:
        print(k, filter_results_0[k])
        print('\n')
    """

from re_model_clustering import ModelClustering
from leep_nlp import leep_nlp
from leep_cv import leep_cv
from op.model import DNNModel, ImageClassModel
import copy
import numpy as np
import math
import os

PATH = os.path.dirname(__file__)

class CoarseRecall:
    def __init__(self, task, data, PERF_MATRIX, model_record, task_type='NLP'):
        self.task_type = task_type
        self.task = task
        self.data = data

        self.model_clustering_instance = ModelClustering(PERF_MATRIX)
        if self.task_type == 'NLP':
            self.model_clustering_result = self.model_clustering_instance.do_cluster()
        if self.task_type == 'CV':
            self.model_clustering_result = self.model_clustering_instance.do_cluster(threshold=0.05)
        self.model_name = self.model_clustering_instance.models
        self.model_record = model_record
        self.cluster_representatives = []
        self.get_cluster_representatives()
        """
        for item in self.cluster_representatives:
            print(self.model_name[item])
        """
        self.label_num = 0
        for item in self.data:
            self.label_num = max(self.label_num, item['label']+1) 
        
        self.leep_flag = False
        self.init_leep(self.label_num)
        self.leep_exp = [math.exp(float(leep_score)) for leep_score in self.leep]


    def init_leep(self, label_num):
        if self.task == "mnli":
            """
            Test Content
            """
            with open(PATH + '/leep_score_mnli.txt', 'r') as f:
                lines = f.readlines()
                self.leep = lines[1].split('\t')
                self.leep_flag = True
        else:
            """
            Calculate LEEP online
            """
            self.leep = [0 for item in self.model_name]
            for item in self.cluster_representatives:
                """
                Get representatives' name.
                Search at pre model table.
                Load model and calculate leep.
                """
                model_name = self.model_name[item]
                record = []
                leep_score = 0
                for temp in self.model_record:
                    if temp['name'] == model_name:
                        record = [temp['name'], 'training_dataset', temp['path'], 'task']
                if self.task_type == 'NLP':
                    model = DNNModel(record, self.label_num)
                    leep_score = leep_nlp(model.model, model.tokenizer, self.data, label_num)
                if self.task_type == 'CV':
                    model = ImageClassModel(record, self.label_num)
                    leep_score = leep_cv(model.model, model.image_processor, self.data, label_num)
                model.unload_model()
                for cluster in self.model_clustering_result:
                    if item in cluster:
                        for index in cluster:
                            self.leep[index] = leep_score
                        

    def get_cluster_representatives(self):
        model_score_avg = copy.deepcopy(self.model_clustering_instance.model_score_avg)
        for cluster in self.model_clustering_result:
            # select representatives for each none singleton cluster
            if len(cluster) > 1:
                cluster_representative = -1
                cluster_max_score = 0
                for item in cluster:
                    if cluster_representative == -1:
                        cluster_representative = item
                        cluster_max_score = model_score_avg[item]
                    else:
                        if model_score_avg[item] > cluster_max_score:
                            cluster_representative = item
                            cluster_max_score = model_score_avg[item]
                self.cluster_representatives.append(cluster_representative)

    def recall(self):
        # step 1: None singleton recall
        # print("---------None Singleton Recall----------")
        model_score_avg = copy.deepcopy(self.model_clustering_instance.model_score_avg)
        proxy_score = np.zeros(len(self.model_name)).tolist()
        for cluster in self.model_clustering_result:
            if len(cluster) > 1:
                for item in cluster:
                    proxy_score[item] = model_score_avg[item] * self.leep_exp[item]

        #print("----result----")

        # step 2: Singleton recall
        #print("-----------Singleton Recall------------")
        for cluster in self.model_clustering_result:
            if len(cluster) == 1:
                curr_model_idx = cluster[0]
                curr_model_acc = np.array(self.model_clustering_instance.model_scores[curr_model_idx])
                curr_model_score = 0
                print(len(self.cluster_representatives))
                for item in self.cluster_representatives:
                    representative_acc = np.array(self.model_clustering_instance.model_scores[item])
                    cos_similarity = curr_model_acc.dot(representative_acc) / (
                            np.linalg.norm(curr_model_acc) * np.linalg.norm(representative_acc))
                    curr_model_score += (cos_similarity * self.leep_exp[item])
                curr_model_score /= len(self.cluster_representatives)
                proxy_score[curr_model_idx] = curr_model_score * model_score_avg[curr_model_idx]

        recall_result = []
        selected_cnt = 15
        print("proxy score:")
        for i in range(len(proxy_score)):
            print("%s: %s" % (self.model_name[i], str(proxy_score[i])))
        while selected_cnt > 0:
            max_idx = proxy_score.index(max(proxy_score))
            recall_result.append(self.model_name[max_idx])
            """
            print("%d: %s selected by proxy score: %s" % (
                max_idx, self.model_name[max_idx], str(proxy_score[max_idx])))
            """
            selected_cnt -= 1
            proxy_score[max_idx] *= -1

        return recall_result


if __name__ == "__main__":
    coarse_recall = CoarseRecall('mnli', None)
    recall_result = coarse_recall.recall()
    print(recall_result)

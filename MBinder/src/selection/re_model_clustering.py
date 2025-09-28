import math
import os
import numpy as np
import random

"""
This file is a re-write version for OLML-postgres.
It should be available for any target tables (that fine-tune model into classifier)
This file calculates LEEP score online. 
"""

PATH = os.path.dirname(__file__)

class ModelClustering:
    datasets = None
    models = None
    model_scores = None
    model_scores_norm = []
    model_score_avg = None
    model_clusters = []
    model_clusters_k_means = []
    max_dataset_scores = []
    min_dataset_scores = []

    def __init__(self, PERF_MATRIX):
        """
        PERF_MATRIX is a plpy select query result.
        It shapes like [{},{},...{}]
        Each dict(row) is a model
        """
        self.datasets = list(PERF_MATRIX[0].keys())[1:]
        self.models = [item['model_name'] for item in PERF_MATRIX]
        self.model_scores = []
        self.max_dataset_scores = [0 for i in range(len(self.datasets))]
        self.min_dataset_scores = [1 for i in range(len(self.datasets))]
        # todo

        for model_info in PERF_MATRIX:
            scores = []
            cnt = 0
            for k, v in model_info.items():
                if k != 'model_name':
                    scores.append(float(v))
                    self.max_dataset_scores[cnt] = max(v, self.max_dataset_scores[cnt])
                    self.min_dataset_scores[cnt] = min(v, self.min_dataset_scores[cnt])
                    cnt += 1
            self.model_scores.append(scores)

        self.model_score_avg = [sum(x) / len(x) for x in self.model_scores]
        self.model_clusters = []
        for i in range(len(self.model_scores)):
            self.model_clusters.append([i])

    def sim_score_by_max_avg_error(self, vec1, vec2):
        errors = []
        for i in range(len(vec1)):
            if vec1[i] > 0 and vec2[i] > 0:
                error = math.fabs(vec1[i] - vec2[i])
                errors.append(error)

        errors = sorted(errors, reverse=True)
        num = min(5, len(errors))
        total_error = 0.0
        for i in range(num):
            total_error = total_error + errors[i]
        return total_error / num

    def cluster_sim(self, cluster_i, cluster_j):
        sum_score = 0.0
        for i in cluster_i:
            for j in cluster_j:
                sum_score = sum_score + self.sim_score_by_max_avg_error(self.model_scores[i], self.model_scores[j])
        return sum_score / (len(cluster_i) * len(cluster_j))

    def do_cluster(self, threshold=0.09):
        while True:
            min_score = 10000
            min_index_i, min_index_j = -1, -1
            for i in range(len(self.model_clusters)):
                cluster_i = self.model_clusters[i]
                for j in range(i + 1, len(self.model_clusters)):
                    cluster_j = self.model_clusters[j]
                    sim_score = self.cluster_sim(cluster_i, cluster_j)
                    if sim_score < min_score:
                        min_index_i, min_index_j = i, j
                        min_score = sim_score

            if min_score <= threshold:
                self.model_clusters[min_index_i] = self.model_clusters[min_index_i] + self.model_clusters[min_index_j]
                del self.model_clusters[min_index_j]

            else:
                break

        # print("clustering result:")
        total_non_singleton_cluster = 0
        total_size = 0
        for i in range(len(self.model_clusters)):
            if len(self.model_clusters[i]) == 1:
                continue
            model_name = ''

            model_and_score = []
            for j in range(len(self.model_clusters[i])):
                model_and_score.append((self.model_clusters[i][j], self.model_score_avg[j]))
            model_and_score = sorted(model_and_score, key=lambda x: x[1], reverse=True)

            for j in range(len(model_and_score)):
                model_name = model_name + ', ' + self.models[model_and_score[j][0]] + ' : ' + str(model_and_score[j][1])
                # model_name = model_name + ', ' + models[model_and_score[j][0]]
            total_non_singleton_cluster = total_non_singleton_cluster + 1
            total_size = total_size + len(self.model_clusters[i])
            #print('model cluster %d, cluster_size: %d, models: %s' % (i, len(self.model_clusters[i]), model_name))

        #print("Total non-singleton cluster num: %d" % total_non_singleton_cluster)
        #print("Total non-singleton cluster size: %d" % total_size)
        #print("Total singleton cluster num: %d" % (len(self.model_clusters) - total_non_singleton_cluster))
        return self.model_clusters

    def do_k_means_cluster(self):
        # set k as the cluster number that Hierarchical Clustering get
        k = len(self.model_clusters)

        # random select k points as origin
        _list = range(len(self.model_scores))
        _slice = random.sample(_list, k)
        center = [] # center has the vector of k centers
        for i in _slice:
            center.append(self.model_scores[i])

        epsilon = 0.00001
        change = 1
        while change > epsilon:
            new_center = []
            new_cluster = []
            for i in range(k):
                new_cluster.append([])

            # get each point into the closest center
            for i in range(len(self.model_scores)):
                closest_center = 0
                closest_avg_error = 1
                for j in range(len(center)):
                    _avg_error = self.sim_score_by_max_avg_error(self.model_scores[i], center[j])
                    if _avg_error < closest_avg_error:
                        closest_avg_error = _avg_error
                        closest_center = j
                new_cluster[closest_center].append(i)
            self.model_clusters_k_means = new_cluster

            # calculate new center and the distance change of the new center
            change = 0
            for i in range(len(center)):
                _arr = []
                for j in range(len(new_cluster[i])):
                    _arr.append(self.model_scores[new_cluster[i][j]])
                _new_center = np.mean(_arr, axis=0)
                change += np.linalg.norm(np.array(_new_center) - np.array(center[i]))
                center[i] = _new_center

        print("clustering result:")
        total_non_singleton_cluster = 0
        total_size = 0
        for i in range(len(self.model_clusters_k_means)):
            if len(self.model_clusters_k_means[i]) == 1:
                continue
            model_name = ''

            model_and_score = []
            for j in range(len(self.model_clusters_k_means[i])):
                model_and_score.append((self.model_clusters_k_means[i][j], self.model_score_avg[j]))
            model_and_score = sorted(model_and_score, key=lambda x: x[1], reverse=True)

            for j in range(len(model_and_score)):
                model_name = model_name + ', ' + self.models[model_and_score[j][0]] + ' : ' + str(
                    model_and_score[j][1])
                # model_name = model_name + ', ' + models[model_and_score[j][0]]
            total_non_singleton_cluster = total_non_singleton_cluster + 1
            total_size = total_size + len(self.model_clusters_k_means[i])
            print(
                'model cluster %d, cluster_size: %d, models: %s' % (i, len(self.model_clusters_k_means[i]), model_name))

        print("Total non-singleton cluster num: %d" % total_non_singleton_cluster)
        print("Total non-singleton cluster size: %d" % total_size)
        print("Total singleton cluster num: %d" % (len(self.model_clusters_k_means) - total_non_singleton_cluster))

    def silhouette_coefficient(self):
        sc = []
        # first get distance matrix for calculation
        distance_matrix = np.zeros((len(self.models), len(self.models)))
        for i in range(len(self.models)):
            for j in range(i):
                model_score_x = self.model_scores[i]
                model_score_y = self.model_scores[j]
                #_x = np.array(model_score_x)
                #_y = np.array(model_score_y)
                # distance_matrix[i][j] = distance_matrix[j][i] = np.linalg.norm(_x - _y)
                distance_matrix[i][j] = distance_matrix[j][i] = self.sim_score_by_max_avg_error(model_score_x, model_score_y)
        #print(distance_matrix)

        flatten_cluster_member = [i for arr in self.model_clusters for i in arr]
        for i in range(len(self.model_clusters)):
            cluster_sc = []
            #print(self.model_clusters[i])
            if len(self.model_clusters[i]) > 1:
                for item in self.model_clusters[i]:
                    # get average distance within cluster _a
                    _a = 0
                    for other_item in self.model_clusters[i]:
                        _a += distance_matrix[item][other_item]
                    _a /= (len(self.model_clusters[i]) - 1)

                    # get the average distance not in cluster _b
                    _b = 10000
                    _total = 0
                    _cnt = 0
                    for j in range(len(self.models)):
                        if j in flatten_cluster_member:
                            if j not in self.model_clusters[i]:
                                _total += distance_matrix[item][j]
                                _cnt += 1
                                # _b = min(_b, distance_matrix[item][j])
                    _b = _total / _cnt
                    _result = (_b - _a) / max(_a, _b)
                    sc.append(_result)
                    cluster_sc.append(_result)
                cluster_sc_result = np.mean(np.array(cluster_sc))
                print("cluster silhouette_coefficient: %s" % str(cluster_sc_result))
        sc_result = np.mean(np.array(sc))
        print("silhouette_coefficient: %s" % str(sc_result))
        return sc_result

    def cluster_model_similarity(self):
        avg_cluster_similarity = 0
        cnt = 0
        for model_score in self.model_scores:
            model_score_norm = (np.array(model_score) - np.array(self.min_dataset_scores)) / (
                                    np.array(self.max_dataset_scores) - np.array(self.min_dataset_scores))
            self.model_scores_norm.append(model_score_norm)
        #print(self.model_scores_norm)
        with open("model_scores_norm.txt", "w") as f:
            for score_norm in self.model_scores_norm:
                norm = []
                for score in score_norm:
                    norm.append(str(score))
                str_norm = "\t".join(norm) + "\n"
                f.write(str_norm)

        for cluster in self.model_clusters:
            if len(cluster) > 1:
                similarities = []
                for i in range(len(cluster)):
                    for j in range(i):
                        similarity = self.sim_score_by_max_avg_error(self.model_scores[cluster[i]], self.model_scores[cluster[j]])
                        similarities.append(similarity)
                similarity_avg = 1 - np.mean(np.array(similarities))
                avg_cluster_similarity += similarity_avg
                cnt += 1
                print("cluster similarity: %s" % str(similarity_avg))
        avg_cluster_similarity = avg_cluster_similarity / cnt
        print("average cluster similarity: %s" % str(avg_cluster_similarity))

    def cluster_model_similarity_test(self):
        for cluster in self.model_clusters:
            if len(cluster) > 1:
                similarities = []
                for i in range(len(cluster)):
                    for j in range(i):
                        max_avg_error = self.sim_score_by_max_avg_error(self.model_scores[cluster[i]], self.model_scores[cluster[j]])
                        similarity = 1 - max_avg_error
                        similarities.append(similarity)
                similarity_avg = np.mean(np.array(similarities))
                print("cluster similarity: %s" % str(similarity_avg))

    def cluster_model_similarity_k_means(self):
        _ = self.model_clusters
        self.model_clusters = self.model_clusters_k_means
        self.cluster_model_similarity()
        self.model_clusters = _

    def silhouette_coefficient_k_means(self):
        # calculate silhouette coefficient of k_means result
        _ = self.model_clusters
        self.model_clusters = self.model_clusters_k_means
        self.silhouette_coefficient()
        self.model_clusters = _

if __name__ == '__main__':
    model_cluster_instance = ModelClustering()
    #model_cluster_instance.do_cluster()
    #model_cluster_instance.cluster_model_similarity()
    #model_cluster_instance.silhouette_coefficient()
    #model_cluster_instance.cluster_model_similarity_test()
    #print('##############################')
    #model_cluster_instance.do_k_means_cluster()
    #model_cluster_instance.cluster_model_similarity_k_means()
    #model_cluster_instance.silhouette_coefficient_k_means()


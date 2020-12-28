from itertools import cycle

from sklearn.cluster import DBSCAN, AffinityPropagation, KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import numpy as np
import sklearn
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances_argmin


class Tracer:
    def __init__(self, df):
        self.epsilon = 0.0018288  # a radial distance of 6 feet in kilometers
        self.data = df

    def get_infected_people(self, patient):
        patient_clusters = []
        model = DBSCAN(eps=self.epsilon, min_samples=2, metric='euclidean').fit(self.data[['latitude', 'longitude']])

        centers = [self.data[['latitude', 'longitude']]]
        centriods = centers[0].values
        n_clusters = len(centriods)
        X, labels_true = make_blobs(n_samples=100, centers=centriods, cluster_std=self.epsilon,
                                    random_state=0)
        k_means = KMeans(init='k-means++', n_clusters=15, n_init=10)
        k_means.fit(X)
        # clusters  = model[0]

        k_means_cluster_centers = k_means.cluster_centers_
        k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)


        # masks = [True if ]

        fig = plt.figure(figsize=(8, 3))
        colors = ['#4EACC5', '#FF9C34']

        for k, col in zip(range(n_clusters), colors):
            my_members = k_means_labels == k
            cluster_center = k_means_cluster_centers[k]
            plt.scatter(X[my_members, 0], X[my_members, 1])
            plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                     markeredgecolor='k', markersize=8)
            # sns.scatterplot(X[my_members, 0], X[my_members, 1], hue=['cluster-{}'.format(x) for x in my_members])
        plt.legend(bbox_to_anchor=[1, 1])
        plt.show()

        # self.data['cluster'] = model.labels_.tolist()
        # core_samples_mask = np.zeros_like(model.labels_, dtype=bool)
        # core_samples_mask[model.core_sample_indices_] = True
        # labels = model.labels_
        # #
        # # # Number of clusters in labels, ignoring noise if present.
        # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        # n_noise_ = list(labels).count(-1)
        # #
        # print('Estimated number of clusters: %d' % n_clusters_)
        # print('Estimated number of noise points: %d' % n_noise_)
        # fig = plt.figure(figsize=(8, 3))
        # sns.scatterplot(self.data['latitude'], self.data['longitude'], hue=['cluster-{}'.format(x) for x in labels])
        # plt.legend(bbox_to_anchor=[1, 1])
        # plt.show()

        # for i in range(len(self.data)):
        #     print(self.data['id'][i])
        #     if self.data['id'][i] == patient:
        #         if self.data['id'][i] not in patient_clusters:
        #             patient_clusters.append(self.data['id'][i])
        # infected_patients = []
        # print(patient_clusters)
        # for cluster in patient_clusters:
        #     print(cluster)
        #     if cluster != -1:
        #         ids_in_cluster = self.data.loc[self.data['cluster'] == cluster, 'id']
        #         for i in range(len(ids_in_cluster)):
        #             patient_id = ids_in_cluster.iloc[i]
        #             if (patient_id not in infected_patients) and (patient_id != patient):
        #                 infected_patients.append(patient_id)
        #             else:
        #                 continue
        # return infected_patients

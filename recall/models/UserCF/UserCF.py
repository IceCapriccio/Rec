"""
2022.5.14
UserCF model
N=10, K=10
recall rate: 2.852%
"""
import warnings
import pandas as pd
warnings.filterwarnings("ignore")
import torch
from torch import nn
import torchhandle
import numpy as np
from recall import utils
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import heapq


class ItemCFRecall(nn.Module):
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, N: int = 10, K: int = 20):
        """
        :param N: history last N  movie
        :param K: Recall@K.  K <= N * N
        """
        super(ItemCFRecall, self).__init__()
        self.train_df = train_df
        self.test_df = test_df
        self.N = N
        self.K = K
        self.nusers = len((self.train_df['userId']).unique())
        self.nmovies = len(pd.concat([self.train_df['movieId'], self.test_df['movieId']]).unique())

        self.ui = np.zeros((self.nusers + 1, self.nmovies + 1))  # (users * movies)
        self.LastN = None  # user -> last N item that user interacted with
        self.SimN = None  # item -> N most similar items
        self.rec = None  # recommendation list for each user, including K items in each list
        self.truth = None  # really choose movies for each user

        self._build_ui()
        self._build_LastN_index(self.N)
        self._build_MostSimN_index(self.N)

    def _build_LastN_index(self, N: int):
        """
        建立每一个用户到最近交互的N个物品的索引
        """
        self.LastN = [[] for userId in range(self.nusers + 1)]
        for userId, user_df in self.train_df.groupby('userId'):
            self.LastN[userId] = user_df.sort_values('timestamp', ascending=False)[:N]['movieId'].tolist()

    def _build_ui(self):
        """
        建立user-item矩阵(nusers * nmovies)
        """
        for index, (userId, movieId, rating, *_) in self.train_df.iterrows():
            self.ui[userId][movieId] = rating

    def _build_MostSimN_index(self, N: int):
        """
        建立每个用户到与其最相似的N个用户的索引
        最耗时的一步
        """
        users_sim = [[] for movieId in range(self.nusers + 1)]
        sim = cosine_similarity(self.ui)
        for i in range(1, self.nusers + 1):
            for j in range(i + 1, self.nusers + 1):
                if len(users_sim[i]) < N:
                    heapq.heappush(users_sim[i], (sim[i, j], j))
                elif users_sim[i][0][0] < sim[i, j]:
                    heapq.heapreplace(users_sim[i], (sim[i, j], j))

                if len(users_sim[j]) < N:
                    heapq.heappush(users_sim[j], (sim[i, j], i))
                elif users_sim[j][0][0] < sim[i, j]:
                    heapq.heapreplace(users_sim[j], (sim[i, j], i))

        self.SimN = users_sim

    def fit(self):
        rec_rating = [{} for userId in range(self.nusers + 1)]
        for userId in range(1, self.nusers + 1):
            mostSimUsers = self.SimN[userId]
            for sim, simUser in mostSimUsers:
                lastN = self.LastN[simUser]
                for item in lastN:
                    rating = sim * self.ui[simUser][item]
                    rec_rating[userId][item] = rec_rating[userId].get(item, 0) + rating

        rec_list = [[] for userId in range(self.nusers + 1)]
        for userId in range(1, self.nusers + 1):
            user_rec = sorted(list(rec_rating[userId].items()), key=lambda x: x[1], reverse=True)[:self.K]
            rec_list[userId].extend(user_rec)

        self.rec = rec_list

    def test(self, K: int):
        """
        metric: micro-Recall@K
        """
        hits = 0
        total_recs = sum([len(each) for each in self.rec])
        self.truth = [[] for userId in range(self.nusers + 1)]

        # cal truth
        for index, (userId, movieId, rating, *_) in self.test_df.iterrows():
            self.truth[userId].append(movieId)

        for userId in range(1, self.nusers + 1):
            hits += len(set([each[0] for each in self.rec[userId]]) & set(self.truth[userId]))

        recall = hits / total_recs
        print(f"recall rate: {recall}")


if __name__ == '__main__':
    train_df, test_df = utils.split_latestN(N=10)
    userCF = ItemCFRecall(train_df, test_df)
    userCF.fit()
    userCF.test(K=10)



"""
2022.5.14
ItemCF model
N=10, K=10
recall rate: 3.990%
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
        建立每个物品到与其最相似的N个物品的索引
        最耗时的一步
        """
        movies_dis = [[] for movieId in range(self.nmovies + 1)]
        sim = cosine_similarity(self.ui.T)
        for i in tqdm(range(1, self.nmovies + 1)):
            for j in range(i + 1, self.nmovies + 1):
                if len(movies_dis[i]) < N:
                    heapq.heappush(movies_dis[i], (sim[i, j], j))
                elif movies_dis[i][0][0] < sim[i, j]:
                    heapq.heapreplace(movies_dis[i], (sim[i, j], j))

                if len(movies_dis[j]) < N:
                    heapq.heappush(movies_dis[j], (sim[i, j], i))
                elif movies_dis[j][0][0] < sim[i, j]:
                    heapq.heapreplace(movies_dis[j], (sim[i, j], i))

        self.SimN = movies_dis

    def fit(self):
        rec_rating = [{} for userId in range(self.nusers + 1)]
        for userId in range(1, self.nusers + 1):
            lastn = self.LastN[userId]
            for movieId in lastn:
                for sim, simMovieId in self.SimN[movieId]:
                    rating = self.ui[userId][movieId] * sim
                    rec_rating[userId][simMovieId] = rec_rating[userId].get(simMovieId, 0) + rating

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



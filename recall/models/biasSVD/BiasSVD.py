"""
Regularized FunkSVD
效果较差，Recall@10基本没有超过1%的
"""
import warnings
import pandas as pd
warnings.filterwarnings("ignore")
import torch
from torch import nn
import torchhandle
from torchhandle.workflow import Metric, BaseContext, Session
import numpy as np
from recall import utils
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from typing import List
from torch.utils.data import Dataset, DataLoader


class BiasSVD(nn.Module):
    def __init__(self, nusers: int, nmovies: int, hid_size: int = 128):
        super(BiasSVD, self).__init__()
        self.user_embd = nn.Embedding(nusers, hid_size)
        self.movie_embd = nn.Embedding(nmovies, hid_size)

    def forward(self, ui: torch.Tensor):
        users, movies = ui[:, 0], ui[:, 1]
        user_embd = self.user_embd(users)  # (bs, hid)
        movie_embd = self.movie_embd(movies)  # (bs, hid)
        ratings = (user_embd * movie_embd).sum(-1)  # (bs)
        return ratings


class BiasSVDDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        super(BiasSVDDataset, self).__init__()
        self.df = df

    def __getitem__(self, item):
        row = self.df.iloc[item]
        user, movie, rating, *_ = row
        return torch.tensor([user, movie], dtype=torch.long), torch.tensor(rating, dtype=torch.float)

    def __len__(self):
        return len(self.df)


def cal_recallk(user_embd: nn.Embedding, movie_embd: nn.Embedding, valid_df, K=10):
    hits = 0
    total = valid_df.shape[0]

    ratings = user_embd @ movie_embd.T
    values, indices = ratings.topk(K)

    for userId, userDf in valid_df.groupby('userId'):
        rec = userDf['movieId'].to_list()
        truth = indices[userId].tolist()
        hit_movies = set(rec) & set(truth)
        hits += len(hit_movies)
        for movie in hit_movies:
            print(userId, movie)
    return hits / total


if __name__ == '__main__':
    data_path = f'../../../data/MovieLens-1M/ratings.csv'
    train_df, test_df = utils.split_latestN(data_path)
    train_ds, test_ds = BiasSVDDataset(train_df), BiasSVDDataset(test_df)
    train_dl, test_dl = DataLoader(train_ds, batch_size=128, shuffle=True), DataLoader(test_ds, batch_size=128)
    device = 'cpu' if torch.cuda.is_available() else 'cuda'

    nusers = pd.concat([train_df, test_df])['userId'].unique().size + 1
    nmovies = pd.concat([train_df, test_df])['movieId'].unique().size + 1
    model = BiasSVD(nusers=nusers, nmovies=nmovies)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    # bar = tqdm(range(100))
    for epoch in range(100):
        model.train()
        trn_loss = 0.
        for x, y in train_dl:
            output = model(x)
            loss = criterion(y, output) + 0.1 * (torch.norm(model.user_embd(x[:, 0]), p=2, dim=-1).sum() + \
                                                 torch.norm(model.movie_embd(x[:, 1]), p=2, dim=-1).sum())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            trn_loss += loss.cpu().detach().item()

        model.eval()
        recallk = cal_recallk(model.user_embd(torch.tensor(range(nusers))), model.movie_embd(torch.tensor(range(nmovies))), test_df)
        print(f"epoch: {epoch}, train loss: {trn_loss: .5f}, Recall@K: {recallk: .5%}")





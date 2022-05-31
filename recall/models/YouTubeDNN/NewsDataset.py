import random

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch import Tensor, nn, tensor
import numpy as np
from collections import Counter
from typing import List, Tuple
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from pathlib import Path
pd.set_option('expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 1000)
pd.set_option('max_colwidth', 200)


class NewsDataset(Dataset):
    """
    一个样本的特征由train_data和user_info合并得到。
        用户id：Sparse
        历史观看视频id：Sparse，List[物品id]，average Embedding
        操作系统：Dense，2类别，编码为0/1，直接输入
        省：Sparse，300+类别
        市：Sparse,700+类别
        年龄(age**)：Dense，年龄范围概率，0-1之间实数
        性别(male/female)：Dense, 连续特征性别概率，0-1之间实数
        网络环境：Dense，4个类别，One-Hot特征当做浮点数输入
        刷新次数(refresh**)：Dense, 0-644之间整数，归一化至[0, 1]；分为8个桶输入
        展现位置(showPos**)：Dense, 0-2698之间整数，归一化至[0, 1]；分为8个桶输入
        物品id：Sparse，Embedding
        是否点击：0/1，标签信息

        unique users = 919603
        unique items = 306661
    """
    def __init__(self, df: pd.DataFrame):
        super(NewsDataset, self).__init__()
        print('preparing dataset...')
        self.sparse = tensor(df[['userId', 'deviceName', 'OS', 'province', 'city']].values.astype('int32'))
        dense = df[['refresh', 'showPos', '0-24', '25-29', '30-39', '40-', 'male', 'female'] +
                        [f'network_{i}' for i in [2, 3, 4, 5]] + [f'refresh{i}' for i in range(0, 8)] +
                        [f'showPos{i}' for i in range(0, 8)]]
        self.dense = tensor(dense.values.astype('float32'))
        self.items = tensor(df[['itemId', 'neg1', 'neg2', 'neg3']].values.astype('int64'))
        self.itemHist = tensor(np.stack(df.itemHist.values, axis=0).astype('int64'))
        del df

    def __getitem__(self, item: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:  # disc, cont, itemId, click
        return self.sparse[item], self.dense[item], self.itemHist[item], self.items[item]

    def __len__(self):
        return self.df.shape[0]


def transform(df: pd.DataFrame):
    """
    label encode and normalize
    :param df:
    :return:
    """
    df.drop('duration', inplace=True, axis=1)  # duration is unused in matching

    df.showTime = df.showTime.astype('int')

    encode_features = ['userId', 'itemId', 'network', 'deviceName', 'OS', 'province', 'city']
    normalize_features = ['refresh', 'showPos']
    for feat in encode_features:
        le = LabelEncoder()
        df[feat] = le.fit_transform(df[feat])
    df['itemId'] = df['itemId'] + 1

    df.refresh = df.refresh.astype('int16').clip(upper=10)
    df.showPos = df.showPos.astype('int16').clip(upper=100)
    for feat in normalize_features:
        scaler = MinMaxScaler()
        df[feat] = scaler.fit_transform(df[feat].to_numpy().reshape(-1, 1))

    return df


def gen_data(data: pd.DataFrame, neg_per_pos: int = 3, pos_per_user: int = 3, hist_len: int = 10):
    n = data.shape[0]
    count = Counter(data.itemId)

    # constructing negative samples
    print('constructing negative samples...')
    neg_num = data.shape[0] * neg_per_pos  # total number of negative samples
    items, p = [], []  # all uniques items and possibility
    for item, cnt in count.items():
        items.append(item)
        p.append(cnt / n)
    neg_items = np.random.choice(items, size=neg_num, replace=True, p=p)
    neg = []
    for i in range(data.shape[0]):
        neg.append(neg_items[i * neg_per_pos: (i + 1) * neg_per_pos])
    data[['neg1', 'neg2', 'neg3']] = neg

    # construct history
    print('constructing history...')
    train, test = [], []
    for userId, userHist in tqdm(data.groupby('userId')):
        itemList = userHist.itemId.tolist()
        userHistLen = len(itemList)
        if userHistLen < pos_per_user:
            continue
        userHist.sort_values('showTime', inplace=True)
        hist = [itemList[max(0, i - hist_len): i] + [0] * (hist_len - i) for i in range(userHistLen)]
        userHist['itemHist'] = hist
        train.append(userHist.iloc[:-1])
        test.append(userHist.iloc[-1])
    print('concatenating...')
    del data
    train = pd.concat(train, axis=0)
    test = pd.concat(test, axis=1).T

    return train, test


if __name__ == '__main__':
    data_dir = Path('~/data')
    data = pd.read_pickle('~/data/clicks.pkl')
    print(data.head())
    data = transform(data)
    # data.to_pickle('~/data/transformed.pkl')
    # data = pd.read_pickle('~/data/transformed.pkl')
    print(data.head())
    train_df, test_df = gen_data(data, neg_per_pos=3)
    # train_df, test_df = pd.read_pickle('~/data/train.pkl'), pd.read_pickle('~/data/test.pkl')
    test_ds = NewsDataset(test_df)
    train_ds = NewsDataset(train_df)
    print(train_ds[0])


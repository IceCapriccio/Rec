import random

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch import Tensor, nn, tensor
import numpy as np
from typing import List, Tuple
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder


class NewsDataset(Dataset):
    """
    一个样本的特征由train_data和user_info合并得到。
        用户id：离散特征，Embedding
        历史观看视频id：离散特征，List[物品id]，average Embedding
        操作系统：离散特征，2类别，编码为0/1，直接输入
        省：离散特征，300+类别，Embedding
        市：离散特征,700+类别，Embedding
        年龄(age**)：连续特征，年龄范围概率，0-1之间实数，直接输入
        性别(male/female)：连续特征性别概率，0-1之间实数，直接输入
        网络环境：离散特征，4个类别，One-Hot特征输入
        刷新次数(refresh**)：0-644之间整数，分桶输入
        展现位置(showPos**)：0-2698之间整数，分桶输入
        物品id：离散特征，Embedding
        是否点击：0/1，标签信息

    unique item IDs: 218364
    """
    def __init__(self, df: pd.DataFrame):
        super(NewsDataset, self).__init__()
        print('preparing dataset...')
        self.df = df

        discFeat = tensor(df[['userId', 'province', 'city']].values, dtype=torch.int)

        # process hist column in string object
        data = []
        for s in tqdm(df['hist']):
            s = s.strip('\'[]').split(', ')
            if s[0] == '':
                s = []
            else:
                s = list(map(int, s))
            s.extend([-1] * (10 - len(s)))
            data.append(s)
        hist = tensor(data, dtype=torch.int) + 1
        self.discFeat = torch.cat([discFeat, hist], dim=1)

        self.contFeat = tensor(df[['OS', 'age024', 'age2529', 'age3039', 'age40', 'male', 'female', 'showPos025',
                                    'showPos2550', 'showPos5075', 'showPos57100', 'refresh025', 'refresh2550',
                                'refresh5075', 'refresh57100', 'net_2', 'net_3', 'net_4', 'net_5']].values)  # 19

        self.itemId = tensor(df['itemId'], dtype=torch.int) + 1  # -1 is empty hist, plus 1 to fit embedding index
        self.click = tensor(df['click'], dtype=torch.float)

    def __getitem__(self, item: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:  # disc, cont, itemId, click
        return self.discFeat[item], self.contFeat[item], self.itemId[item], self.click[item]

    def __len__(self):
        return self.df.shape[0]


def gen_data(uiDf: pd.DataFrame, userDf: pd.DataFrame, histLen: int = 10, negNumEachPos: int = 3,
             posNumEachUser: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    :param uiDf:
    :param userDf:
    :param posNumEachUser: 每个用户产生多少正样本
    :param negNumEachPos: 每个正样本对应多少负样本
    :param histLen: 每个样本的历史记录长度
    :return:
    """
    # 筛选出历史交互大于 leastHist的用户
    validDf = []
    for userId, curDf in tqdm(uiDf.groupby('userId')):
        clicks = curDf[curDf.click == '1']
        if clicks.shape[0] >= posNumEachUser + 1 and curDf.shape[0] - clicks.shape[0] >= posNumEachUser * negNumEachPos:  # 还有一个当验证集
            validDf.append(curDf)
    print('concatenating...')
    validDf = pd.concat(validDf, axis=0)
    validDf.userId = validDf.userId.cat.remove_unused_categories()
    validDf.itemId = validDf.itemId.cat.remove_unused_categories()
    print(f'valid df: {validDf.shape}')

    trainSet, evalSet = [], []
    for uId, curDf in tqdm(validDf.groupby('userId')):  # 逐用户生成正样本和负样本 1:3
        curDf = curDf.sort_values(by='showTime', ascending=False)
        clicks = curDf[curDf.click == '1']  # 所有的点击交互

        posSamples = clicks[:clicks.shape[0] - 1].sample(n=posNumEachUser)  # 每个用户产生posNumEachUser个正样本
        negSamples = curDf[curDf.click == '0'].sample(n=posNumEachUser * negNumEachPos)  # 每个正样本产生negNum个负样本

        # userinfo
        userInfo = userDf.loc[uId]

        def gen_sample(uiInfo, userInfo):
            # generate one sample from one user-item row and user info row
            OS = userInfo.OS
            province = userInfo.province
            city = userInfo.city
            age024 = userInfo['0-24']
            age2529 = userInfo['25-29']
            age3039 = userInfo['30-39']
            age40 = userInfo['40-']
            male = userInfo.male
            female = userInfo.female

            showTime = uiInfo.showTime.value
            network = uiInfo.network
            refresh = uiInfo.refresh
            showPos = uiInfo.showPos
            hist = clicks[clicks.showTime.view(int) < showTime]['itemId'].tolist()[:histLen]  # get most $histLen$ items interaction history which earlier than current item
            itemId = uiInfo.itemId
            click = uiInfo.click

            sample = [uId, itemId, hist, OS, province, city, age024, age2529, age3039, age40, male, female, network,
                      refresh, showPos, click]
            return sample

        for index, uiInfo in posSamples.iterrows():  # 剩一个当做验证集
            trainSet.append(gen_sample(uiInfo, userInfo))
        for index, uiInfo in negSamples.iterrows():
            trainSet.append(gen_sample(uiInfo, userInfo))
        uiInfo = clicks.iloc[0]
        uiInfo.click = '2'
        evalSet.append(gen_sample(uiInfo, userInfo))
    columns = ['userId', 'itemId', 'hist', 'OS', 'province', 'city', 'age024', 'age2529', 'age3039', 'age40', 'male', 'female',
               'network', 'refresh', 'showPos', 'click']
    trainDf = pd.DataFrame(data=trainSet, columns=columns)
    evalDf = pd.DataFrame(data=evalSet, columns=columns)
    return trainDf, evalDf


def raw2train_data():
    """
    从csv原始输入转为DataFrame
    :return:
    """
    ui_path = '~/Rec/data/NewsRec/off_data/ui_cut.csv'
    user_path = '~/Rec/data/NewsRec/off_data/user_cut.csv'
    # ui_path = '../../../data/ui_cut.csv'
    # user_path = '../../../data/user_cut.csv'
    ui_df = pd.read_csv(str(ui_path), header=0, parse_dates=['showTime'], index_col=0,
                        dtype={'userId': 'category', 'itemId': 'category', 'showTime': 'object', 'network': 'category',
                               'refresh': 'category', 'showPos': 'category', 'click': 'category', 'duration': 'int32'})
    ui_df.reset_index(drop=True, inplace=True)
    user_df = pd.read_csv(str(user_path), header=0, index_col=0,
                          dtype={'userId': 'category', 'deviceName': 'object', 'OS': 'category',
                                 'province': 'category', 'city': 'category', 'age': 'object', 'gender': 'object'})
    print('read finished.')

    trainDf, evalDf = gen_data(ui_df, user_df)

    return trainDf, evalDf


def transform(df: pd.DataFrame, encoder: dict):
    """
    某些列的分桶、数据类型的转换、稀疏特征的编码、
    :param df:
    :param encoder: raw item ID -> dense continuous ID
    :return:
    """
    trnPos = df.showPos.astype('int')
    trnPosDm = pd.get_dummies(pd.qcut(trnPos, q=[0., .25, .5, .75, 1.],
                                      labels=['showPos025', 'showPos2550', 'showPos5075', 'showPos57100']))
    df = pd.concat([df.drop('showPos', axis=1), trnPosDm], axis=1)

    trnRefresh = df.refresh.astype('int')
    trnRefreDm = pd.get_dummies(pd.qcut(trnRefresh, q=[0., .25, .5, .75, 1.],
                                        labels=['refresh025', 'refresh2550', 'refresh5075', 'refresh57100']))
    df = pd.concat([df.drop('refresh', axis=1), trnRefreDm], axis=1)

    trnNetwork = df.network
    trnNetDm = pd.get_dummies(trnNetwork, prefix='net')
    df = pd.concat([df.drop('network', axis=1), trnNetDm], axis=1)

    df.OS = df.OS.map(lambda x: 1 if x == 'Android' else 0)

    proLe = LabelEncoder()
    df.province = proLe.fit_transform(df.province.values)

    cityLe = LabelEncoder()
    df.city = cityLe.fit_transform(df.city.values)

    df.userId = df.userId.astype('category').cat.codes

    for i, itemId in enumerate(df.itemId):
        if itemId not in encoder:
            encoder[itemId] = len(encoder)
        df.itemId[i] = encoder[itemId]

    hist = df['hist']
    for i, itemList in enumerate(hist):
        for j in range(len(itemList)):
            if itemList[j] not in encoder:
                encoder[itemList[j]] = len(encoder)
            hist[i][j] = encoder[itemList[j]]
    df['hist'] = hist
    print(len(encoder))
    return df


if __name__ == '__main__':
    trainDf, evalDf = raw2train_data()
    encoder = {}  # raw item ID -> dense continuous ID
    trainDf = transform(trainDf, encoder)
    evalDf = transform(evalDf, encoder)
    trainDf.to_csv('train_df.csv')
    evalDf.to_csv('eval_df.csv')
    trainDf = pd.read_csv

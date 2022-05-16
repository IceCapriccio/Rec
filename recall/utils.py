import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas
from typing import Tuple


def split_latestN(data_path: str, N: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    对于每个用户，将每个用户最后交互的N个物品作为预测集，其余作为训练集
    """
    df = pd.read_csv(str(data_path))
    df['time'] = pd.to_datetime(df['timestamp'], unit='s')

    # movieId是不连续的，在这里映射为连续的
    movid_reset_dic = {value: idx + 1 for idx, value in enumerate(df['movieId'].unique())}
    df['movieId'] = df['movieId'].map(movid_reset_dic)

    user_reset_dic = {value: idx + 1 for idx, value in enumerate(df['userId'].unique())}
    df['userId'] = df['userId'].map(user_reset_dic)

    train_df = pd.DataFrame(columns=df.columns)
    test_df = pd.DataFrame(columns=df.columns)

    for userId, user_df in df.groupby('userId'):
        user_df.sort_values(by='timestamp', inplace=True)
        train_df = train_df.append(user_df[:-N])
        test_df = test_df.append(user_df[-N:])

    return train_df, test_df

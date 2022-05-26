"""
Use NewsRec dataset.
metrics are nearly 0.
shit!
"""
import pandas as pd
import torch
from torch import nn, optim
from torch import Tensor
from torch.utils.data import DataLoader
import numpy as np
from NewsDataset import NewsDataset
from tqdm import tqdm
from annoy import AnnoyIndex
from collections import defaultdict


class YouTubeDNN(nn.Module):

    def __init__(self, item_num: int, user_num: int, city_num: int, province_num: int, embedding_dim: int = 16):
        super(YouTubeDNN, self).__init__()

        self.itemEmbedding = nn.Embedding(item_num + 1, embedding_dim, padding_idx=0)  # extra 1 for empty
        self.userEmbedding = nn.Embedding(user_num, embedding_dim)
        self.cityEmbedding = nn.Embedding(city_num, embedding_dim)
        self.provEmbedding = nn.Embedding(province_num, embedding_dim)
        self.lin1 = nn.Linear(83, 128)
        self.lin2 = nn.Linear(128, 32)
        self.lin3 = nn.Linear(32, 16)

    def forward(self, disc: Tensor, cont: Tensor, itemId: Tensor):
        userEmbedding = self.user_tower(disc, cont)  # (bs, emb dim)

        itemEmbedding = self.itemEmbedding(itemId)  # (bs, emb dim)
        logits = (userEmbedding * itemEmbedding).sum(dim=1)  # (bs)
        scores = torch.sigmoid(logits)
        return scores

    def user_tower(self, disc: Tensor, cont: Tensor) -> Tensor:
        """
        concatenate features and go through MLP
        :param disc: userId province city hist(10)
        :param cont:
        :return:
        """
        # disc:
        userId = disc[:, 0]
        province = disc[:, 1]
        city = disc[:, 2]
        hist = disc[:, 3:]

        userId_embedding = self.userEmbedding(userId)
        prov_embedding = self.provEmbedding(province)
        city_embedding = self.cityEmbedding(city)
        hist_embedding = self.itemEmbedding(hist).mean(dim=1)  # kick out empty hist

        userEmbedding = torch.cat([userId_embedding, prov_embedding, city_embedding, hist_embedding, cont], dim=1).float()  # (bs, 83)
        userEmbedding = self.lin1(userEmbedding)
        userEmbedding = self.lin2(userEmbedding)
        userEmbedding = self.lin3(userEmbedding)  # (bs, emb dim)
        return userEmbedding

# following Annoy and topk_metric are copied from torch_rechub
class Annoy(object):
    """
    Vector matching by Annoy
    Args:
        metric (str): distance metric
        n_trees (int): n_trees
        search_k (int): search_k
    """

    def __init__(self, metric='angular', n_trees=10, search_k=-1):
        self._n_trees = n_trees
        self._search_k = search_k
        self._metric = metric

    def fit(self, X):
        self._annoy = AnnoyIndex(X.shape[1], metric=self._metric)
        for i, x in enumerate(X):
            self._annoy.add_item(i, x.tolist())
        self._annoy.build(self._n_trees)

    def set_query_arguments(self, search_k):
        self._search_k = search_k

    def query(self, v, n):
        return self._annoy.get_nns_by_vector(v.tolist(), n, self._search_k, include_distances=True)  #

    def __str__(self):
        return 'Annoy(n_trees=%d, search_k=%d)' % (self._n_trees, self._search_k)


def topk_metrics(y_true, y_pred, topKs=[5]):
    """choice topk metrics and compute it
    the metrics contains 'ndcg', 'mrr', 'recall', 'precision' and 'hit'

    Args:
        y_true (dict): {userid, item_ids}, the key is user id and the value is the list that contains the items the user interacted
        y_pred (dict): {userid, item_ids}, the key is user id and the value is the list that contains the items recommended
        topKs (list or tuple): if you want to get top5 and top10, topKs=(5, 10)

    Return:
        results (dict): {metric_name: metric_values}, it contains five metrics, 'ndcg', 'recall', 'mrr', 'hit', 'precision'

    """
    assert len(y_true) == len(y_pred)

    if not isinstance(topKs, (tuple, list)):
        raise ValueError('topKs wrong, it should be tuple or list')

    pred_array = []
    true_array = []
    for u in y_true.keys():
        pred_array.append(y_pred[u])
        true_array.append(y_true[u])
    ndcg_result = []
    mrr_result = []
    hit_result = []
    precision_result = []
    recall_result = []
    for idx in range(len(topKs)):
        ndcgs = 0
        mrrs = 0
        hits = 0
        precisions = 0
        recalls = 0
        for i in range(len(true_array)):
            if len(true_array[i]) != 0:
                mrr_tmp = 0
                mrr_flag = True
                hit_tmp = 0
                dcg_tmp = 0
                idcg_tmp = 0
                hit = 0
                for j in range(topKs[idx]):
                    if pred_array[i][j] in true_array[i]:
                        hit += 1.
                        if mrr_flag:
                            mrr_flag = False
                            mrr_tmp = 1. / (1 + j)
                            hit_tmp = 1.
                        dcg_tmp += 1. / (np.log2(j + 2))
                    idcg_tmp += 1. / (np.log2(j + 2))
                hits += hit_tmp
                mrrs += mrr_tmp
                recalls += hit / len(true_array[i])
                precisions += hit / topKs[idx]
                if idcg_tmp != 0:
                    ndcgs += dcg_tmp / idcg_tmp
        hit_result.append(round(hits / len(pred_array), 4))
        mrr_result.append(round(mrrs / len(pred_array), 4))
        recall_result.append(round(recalls / len(pred_array), 4))
        precision_result.append(round(precisions / len(pred_array), 4))
        ndcg_result.append(round(ndcgs / len(pred_array), 4))

    results = defaultdict(list)
    for idx in range(len(topKs)):
        output = f'NDCG@{topKs[idx]}: {ndcg_result[idx]}'
        results['NDCG'].append(output)

        output = f'MRR@{topKs[idx]}: {mrr_result[idx]}'
        results['MRR'].append(output)

        output = f'Recall@{topKs[idx]}: {recall_result[idx]}'
        results['Recall'].append(output)

        output = f'Hit@{topKs[idx]}: {hit_result[idx]}'
        results['Hit'].append(output)

        output = f'Precision@{topKs[idx]}: {precision_result[idx]}'
        results['Precision'].append(output)
    return results


def trainer(trainDf: pd.DataFrame, evalDf: pd.DataFrame, epochs=100):
    item_num = 218364
    user_num = trainDf.userId.max() + 1
    province_num = trainDf.province.max() + 1
    city_num = trainDf.city.max() + 1

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trnDs = NewsDataset(trainDf)
    evalDs = NewsDataset(evalDf)
    trnDl = DataLoader(trnDs, batch_size=64, shuffle=True)
    evalDl = DataLoader(evalDs, batch_size=64)
    model = YouTubeDNN(item_num=item_num, user_num=user_num, province_num=province_num, city_num=city_num).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        trn_loss = torch.tensor(0.).to('cuda')
        for disc, cont, itemId, click in tqdm(trnDl, desc=f'Train Epoch {epoch}'):
            # print(disc.shape, cont.shape, itemId.shape, click.shape)
            disc, cont, itemId, click = disc.to(device), cont.to(device), itemId.to(device), click.to(device)
            pred = model(disc, cont, itemId)
            bloss = criterion(pred, click)
            optimizer.zero_grad()
            bloss.backward()
            optimizer.step()
            trn_loss += bloss * disc.shape[0]

        print(f'Loss = {trn_loss.detach().cpu().item()}')

        model.eval()
        annoy = Annoy(n_trees=10)
        itemEmbedding = model.itemEmbedding.weight
        annoy.fit(itemEmbedding)
        topk = 10
        match_res = {}
        ground_truth = {}
        with torch.no_grad():
            for disc, cont, itemId, click in tqdm(evalDl, desc=f'Evaluate Epoch {epoch}'):
                disc, cont, itemId, click = disc.to(device), cont.to(device), itemId.tolist(), click.to(device)
                userEmbedding = model.user_tower(disc, cont)
                userIds = disc[:, 0].tolist()
                for idx in range(len(userIds)):
                    items_idx, items_scores = annoy.query(v=userEmbedding[idx], n=topk)
                    userId = userIds[idx]
                    true_itemId = itemId[idx]
                    match_res[userId] = items_idx
                    ground_truth[userId] = [true_itemId]
        report = topk_metrics(ground_truth, match_res, topKs=[5, 10])
        for key in report:
            print(key, report[key])


if __name__ == '__main__':
    trainDf = pd.read_csv(f'~/Rec/data/NewsRec/train_df.csv', index_col=0)
    evalDf = pd.read_csv(f'~/Rec/data/NewsRec/eval_df.csv', index_col=0)
    print('read finish.')
    trainer(trainDf, evalDf)


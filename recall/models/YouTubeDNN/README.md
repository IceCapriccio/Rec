对YouTubeDNN模型的复现。

数据集采用一点资讯比赛-CTR方向的数据集，每一个正样本对应3个负样本，每个用户采样3个正样本。
每个用户所有点击的时间序的最后一个样本作为验证集。处理好的数据集[下载链接](https://cowtransfer.com/s/03071709d69248)

一个样本的特征由train_data和user_info合并得到。
* 用户id：Sparse 
* 历史观看视频id：Sparse，List[物品id]，average Embedding 
* 操作系统：Dense，2类别，编码为0/1，直接输入 
* 省：Sparse，300+类别 
* 市：Sparse,700+类别
* 年龄(age**)：Dense，年龄范围概率，0-1之间实数 
* 性别(male/female)：Dense, 连续特征性别概率，0-1之间实数 
* 网络环境：Dense，4个类别，One-Hot特征当做浮点数输入 
* 刷新次数(refresh**)：Dense, 0-644之间整数，归一化至[0, 1]；分为8个桶输入 
* 展现位置(showPos**)：Dense, 0-2698之间整数，归一化至[0, 1]；分为8个桶输入
* 物品id：Sparse，Embedding 
* 是否点击：0/1，标签信息

验证集指标采用了torch_rechub的topk_metrics接口，此外也复制了torch_rechub中的Annoy类。

#### 数据集测试结果：
K = 100

Recall@K: 6.22%

Precision@K: 0.06%
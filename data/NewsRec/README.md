一点资讯-CTR：https://tech.yidianzixun.com/competition/#/
第一参赛者笔记：https://www.logicjake.xyz/2021/09/20/%E4%B8%80%E7%82%B9%E8%B5%84%E8%AE%AF%E6%8A%80%E6%9C%AF%E7%BC%96%E7%A8%8B%E5%A4%A7%E8%B5%9BCTR%E8%B5%9B%E9%81%93-%E8%B5%9B%E5%90%8E%E6%80%BB%E7%BB%93/

文件夹中的三篇notebook：
1.1 and 1.2 是对train_data.txt和user_info.txt的EDA和预处理，输出user_item.pkl和user.pkl。（PS：pkl的读取速度是csv的好几倍，所以存储为pkl格式）
2. 将user-item和user-info合并

数据列表：
（1）用户信息user_info.txt，“\t”分割，各列字段为：用户id、设备名称、操作系统、所在省、所在市、年龄、性别；
（2）文章信息doc_info.txt，“\t”分割，各列字段为：文章id、标题、发文时间、图片数量、一级分类、二级分类、关键词；
（3）训练数据train_data.txt，“\t”分割，各列字段为：用户id、文章id、展现时间、网路环境、刷新次数、展现位置、是否点击、消费时长（秒）；
（4）测试数据test_data.txt，“\t”分割，各列字段为：编号、用户id、文章id、展现时间、网路环境、刷新次数；
（5）答案提交示例predict.csv，“,”分割，各列字段为：编号、点击率预估值，该编号与测试数据中的数据一一对应；

数据项说明：
（1）网络环境：0：未知；1：离线；2：WiFi；3：2g；4：3g；5：4g；
（2）刷新次数：用户打开APP后推荐页的刷新次数，直到退出APP则清零；
（3）训练数据取自用户历史12天的行为日志，测试数据采样自第13天的用户展现日志；



'''
1.数据预处理
    数据离散化(x和y都需要进行，一般情况下模型会自动进行处理)
2.建模思想
    决策树、随机森林
    决策树的思想其实就是根据非常多的条件（与离散化处理有关）逐级进行判断并决策
    随机森林的处理，包括记录数量和维度选取
3.求解方式
    纯度：基尼系数、熵、方差
    用纯度来选择分裂维度（可以反过来反映各维度对结果的相关性）

    树层数、叶子数、叶子记录数、分裂点记录数等
    针对上述的各种指标，可以进行前剪枝、后剪枝，防止过拟合
4.模型评估
    准确率
'''
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomTreesEmbedding
from sklearn import tree
from  sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

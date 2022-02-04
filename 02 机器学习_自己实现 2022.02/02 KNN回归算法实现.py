import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

data = pd.read_csv ( r"Iris.csv" )  # 注：用前三个特征回归第四个特征
# 删除不需要的Id与Species列和重复记录
data.drop ( [ "Id" , "Species" ] , axis=1 , inplace=True )  # inplace意为就地删除，不用再赋值给data
data.drop_duplicates ( inplace=True )

t = data.sample ( len ( data ) )
train_X = t.iloc[ :int ( len ( t ) * 0.8 ) , :-1 ]
train_y = t.iloc[ :int ( len ( t ) * 0.8 ) , -1 ]
test_X = t.iloc[ int ( len ( t ) * 0.8 ): , :-1 ]
test_y = t.iloc[ int ( len ( t ) * 0.8 ): , -1 ]


class TrasondKNN2:
    """
    用于回归预测，寻找最近的k个邻居，根据前三个特征属性，根据k个邻居的第四个特征预测当前样本的第四个特征值
    """

    def __init__(self , k):
        """
        :param k: int，邻居的个数
        """
        self.k = k
        self.X = None
        self.y = None

    def fit(self , X , y):
        """
        训练（在KNN方法中实际上不做计算）
        :param X: 类数组类型，形状为[样本数量,特征数量]，待训练的样本特征（属性）
        :param y: 类数组类型，形状为[样本数量]，每个样本的目标值（标签）
        """
        self.X = np.asarray ( X )
        self.y = np.asarray ( y )
        return None

    def predict(self , X):
        """
        根据参数传递的样本，对样本数据进行预测
        :param X: 类数组类型，形状为[样本数量,特征数量]，待预测的样本特征（属性）
        :return: 数组类型，预测的结果
        """
        X = np.asarray ( X )
        result = [ ]
        for x in X:
            dis = np.sqrt ( np.sum ( (x - self.X) ** 2 , axis=1 ) )  # 计算距离
            index = dis.argsort ()  # 排序后原数组索引
            index = index[ :self.k ]  # 找前k个
            result.append ( np.mean ( self.y[ index ] ) )  # 计算均值
        return np.asarray ( result )

    def predict2(self , X):
        """
        考虑权重，权重是邻居距离的倒数 / 所有节点距离倒数之和
        :param X: 类数组类型，形状为[样本数量,特征数量]，待预测的样本特征（属性）
        :return: 数组类型，预测的结果
        """
        X = np.asarray ( X )
        result = [ ]
        for x in X:
            dis = np.sqrt ( np.sum ( (x - self.X) ** 2 , axis=1 ) )  # 计算距离
            index = dis.argsort ()  # 排序后原数组索引
            index = index[ :self.k ]  # 找前k个
            s = np.sum ( 1 / (dis[ index ] + 0.001) )  # 所有邻居距离倒数之和，注意防止分母为0
            weight = (1 / (dis[ index ] + 0.001)) / s  # 计算权重，每个节点距离的倒数除以倒数之和
            result.append ( np.sum ( self.y[ index ] * weight ) )  # 每个标签值乘以对应位置的权重相加
        return np.asarray ( result )


knn = TrasondKNN2 ( k=3 )
knn.fit ( train_X , train_y )
result = knn.predict2 ( test_X )
# print ( "预测结果：\n" , result )
# print ( "正确答案：\n" , test_y.values )
print ( "预测方差：\n" , np.mean ( (result - test_y) ** 2 ) )

mpl.rcParams[ "font.family" ] = "SimHei"
mpl.rcParams[ "axes.unicode_minus" ] = False
plt.figure ( figsize=(10 , 10) )
plt.plot ( result , "ro-" , label="预测值" )
plt.plot ( test_y.values , "go--" , label="真实值" )
plt.title ( "KNN连续值预测展示" )
plt.xlabel ( "节点序号" )
plt.ylabel ( "花瓣宽度" )
plt.legend ()
plt.show ()

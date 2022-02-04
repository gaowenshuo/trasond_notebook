import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

data = pd.read_csv ( r"Order.csv" )
t = data.iloc[ : , -8: ]  # 筛选需要的列


# print(t)

class TrasondKMeans:
    """
    聚类算法
    """

    def __init__(self , k , times):
        """
        :param k:int，聚成几个类
        :param times:int，迭代的次数
        """
        self.k = k
        self.times = times
        self.cluster_centers_ = None
        self.labels_ = None

    def fit(self , X):
        """
        根据提供的训练数据，对模型进行训练
        :param X:类数组类型，形状为[样本数量,特征数量]，待训练的样本特征（属性）
        """
        X = np.asarray ( X )
        # 随机选取k个初始聚类中心
        # np.random.seed () # 设置随机种子，类似random_state
        self.cluster_centers_ = X[ np.random.randint ( 0 , len ( X ) , self.k ) ]
        self.labels_ = np.zeros ( len ( X ) )  # 每个样本的类
        for t in range ( self.times ):  # 每一次迭代
            for index , x in enumerate ( X ):
                dis = np.sqrt ( np.sum ( (x - self.cluster_centers_) ** 2 , axis=1 ) )  # 计算每个样本与聚类中心的距离
                self.labels_[ index ] = dis.argmin ()  # 将最小距离的索引赋值给标签数组，索引的值就是当前点所属的簇，范围为[0，k-1]
            for i in range ( self.k ):  # 循环遍历每一个簇计算均值更新聚类中心
                self.cluster_centers_[ i ] = np.mean ( X[ self.labels_ == i ] , axis=0 )
        return None

    def predict(self , X):
        """
        根据参数传递的样本，对样本数据进行预测属于哪一个簇
        :param:类数组类型，形状为[样本数量,特征数量]，待预测的样本特征（属性）
        :return:数组类型，预测的结果，每一个X所属的类别
        """
        X = np.asarray ( X )
        result = np.zeros ( len ( X ) )
        for index , x in enumerate ( X ):
            dis = np.sqrt ( np.sum ( (x - self.cluster_centers_) ** 2 , axis=1 ) )  # 计算样本到每个聚类中心的距离
            result[ index ] = dis.argmin ()  # 找到最近的聚类中心划分类别
        return result


kmeans = TrasondKMeans ( 3 , 50 )
kmeans.fit ( t )
print ( "聚类中心：\n" , kmeans.cluster_centers_ )
# print ( "属于簇0的数据：\n" , t[ kmeans.labels_ == 0 ] )
# print ( "任意预测：\n" , kmeans.predict ( [ [ 30 , 30 , 40 , 0 , 0 , 0 , 0 , 0 ] , [ 0 , 0 , 0 , 0 , 0 , 30 , 30 , 40 ] ,
#                                        [ 30 , 30 , 0 , 0 , 0 , 0 , 20 , 20 ] ] ) )

# 可视化，只选前两个特征
t2 = data.loc[ : , "Food%":"Fresh%" ]  # loc与iloc不同之处：从标签索引，右闭
kmeans = TrasondKMeans ( 3 , 50 )
kmeans.fit ( t2 )

mpl.rcParams[ "font.family" ] = "SimHei"
mpl.rcParams[ "axes.unicode_minus" ] = False
plt.figure ( figsize=(10 , 10) )
plt.scatter ( t2[ kmeans.labels_ == 0 ].iloc[ : , 0 ] , t2[ kmeans.labels_ == 0 ].iloc[ : , 1 ] , label="类别0" )
plt.scatter ( t2[ kmeans.labels_ == 1 ].iloc[ : , 0 ] , t2[ kmeans.labels_ == 1 ].iloc[ : , 1 ] , label="类别1" )
plt.scatter ( t2[ kmeans.labels_ == 2 ].iloc[ : , 0 ] , t2[ kmeans.labels_ == 2 ].iloc[ : , 1 ] , label="类别2" )
plt.scatter ( kmeans.cluster_centers_[ : , 0 ] , kmeans.cluster_centers_[ : , 1 ] , marker="+" , s=300 )  # 聚类中心
plt.title ( "食物与肉类购买聚类分析" )
plt.xlabel ( "食物" )
plt.ylabel ( "肉类" )
plt.legend ()
plt.show ()

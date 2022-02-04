import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# 读取鸢尾花数据集，header参数：标题的行，默认0，没有就写None
data = pd.read_csv ( r"Iris.csv" , header=0 )
# print ( data )
# print ( "鸢尾花数据集前五行：\n" , data.head ( 5 ) )  # 默认值5
# print ( "鸢尾花数据集后五行：\n" , data.tail ( 5 ) )  # 默认值5
# print ( "鸢尾花数据集随机十行：\n" , data.sample ( 10 ) )  # 默认值1

# 删除不需要的Id列和重复样本，将种类的名字转换为数字
data = data.drop ( "Id" , axis=1 )
data = data.drop_duplicates ()
data[ "Species" ] = data[ "Species" ].map ( {"versicolor": 0 , "setosa": 1 , "virginica": 2} )
# print ( "鸢尾花数据集的样本量为：\n" , len ( data ) )
# print ( "各种类的样本量为：\n" , data[ "Species" ].value_counts () )
# print ( "鸢尾花数据集随机十行：\n" , data.sample ( 10 ) )

# 区分训练集和测试集
# 提取出每个类别的鸢尾花数据
t0 = data[ data[ "Species" ] == 0 ]
t1 = data[ data[ "Species" ] == 1 ]
t2 = data[ data[ "Species" ] == 2 ]
# shuffle
t0 = t0.sample ( len ( t0 ) )  # random_state=...:随机数种子，如果需要每次运行程序保持打乱顺序不变就设置固定的值
t1 = t1.sample ( len ( t1 ) )
t2 = t2.sample ( len ( t2 ) )
train_X = pd.concat (
    [ t0.iloc[ :int ( len ( t0 ) * 0.8 ) , :-1 ] , t1.iloc[ :int ( len ( t1 ) * 0.8 ) , :-1 ] ,
      t2.iloc[ :int ( len ( t2 ) * 0.8 ) , :-1 ] ] ,
    axis=0 )  # 三种类别的并在一起，注意iloc的提取方法
train_y = pd.concat (
    [ t0.iloc[ :int ( len ( t0 ) * 0.8 ) , -1 ] , t1.iloc[ :int ( len ( t1 ) * 0.8 ) , -1 ] ,
      t2.iloc[ :int ( len ( t2 ) * 0.8 ) , -1 ] ] ,
    axis=0 )
test_X = pd.concat (
    [ t0.iloc[ int ( len ( t0 ) * 0.8 ): , :-1 ] , t1.iloc[ int ( len ( t1 ) * 0.8 ): , :-1 ] ,
      t2.iloc[ int ( len ( t2 ) * 0.8 ): , :-1 ] ] ,
    axis=0 )
test_y = pd.concat (
    [ t0.iloc[ int ( len ( t0 ) * 0.8 ): , -1 ] , t1.iloc[ int ( len ( t1 ) * 0.8 ): , -1 ] ,
      t2.iloc[ int ( len ( t2 ) * 0.8 ): , -1 ] ] ,
    axis=0 )


class TrasondKNN:
    """
    K近邻分类方法
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
        self.X = np.asarray ( X )  # 可能是List或者ndarray或者DataFrame，统一转化为ndarray
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

        # ndarray中遍历，每次取数组中的一行
        for x in X:
            dis = np.sqrt ( np.sum ( (x - self.X) ** 2 , axis=1 ) )  # 对于测试集中的每一个样本，依次与训练集中所有样本求欧氏距离，其中利用了ndarray的广播机制
            index = dis.argsort ()  # 不同于sort，返回排序后每个元素在原数组（排序前）中的位置
            index = index[ :self.k ]  # 截断取前k个元素（最近的k个元素的索引）
            count = np.bincount ( self.y[ index ] )  # 返回数组中每个元素出现的次数放在该元素数字号码的位置上，元素必须是非负整数
            result.append ( count.argmax () )  # 返回值最大的元素对应的索引，最大元素索引就是出现次数最多的元素，结果就是我们判定的类别

        return np.asarray ( result )

    def predict2(self , X):
        """
        修改版，考虑权重，更近的邻居权重大，权重使用距离的倒数
        :param X: 类数组类型，形状为[样本数量,特征数量]，待预测的样本特征（属性）
        :return: 数组类型，预测的结果
        """
        X = np.asarray ( X )
        result = [ ]

        # ndarray中遍历，每次取数组中的一行
        for x in X:
            dis = np.sqrt ( np.sum ( (x - self.X) ** 2 , axis=1 ) )  # 对于测试集中的每一个样本，依次与训练集中所有样本求欧氏距离，其中利用了ndarray的广播机制
            index = dis.argsort ()  # 不同于sort，返回排序后每个元素在原数组（排序前）中的位置
            index = index[ :self.k ]  # 截断取前k个元素（最近的k个元素的索引）
            count = np.bincount ( self.y[ index ] , weights=1 / dis[ index ] )  # 返回数组中每个元素的权重的和放在该元素数字号码的位置上，使用weights考虑权重，元素必须是非负整数
            result.append ( count.argmax () )  # 返回值最大的元素对应的索引，最大元素索引就是出现次数最多的元素，结果就是我们判定的类别

        return np.asarray ( result )


knn = TrasondKNN ( k=3 )

# 训练
knn.fit ( train_X , train_y )
# 进行测试，获得测试结果
result = knn.predict2 ( test_X )
print ( "预测结果：\n" , result == test_y )
print ( "\n预测准确率：\n" , np.sum ( result == test_y ) / len ( result ) )

# 可视化
mpl.rcParams[ "font.family" ] = "SimHei"  # 支持中文显示，设置字体为黑体
mpl.rcParams[ "axes.unicode_minus" ] = False  # 中文字体时正常使用负号(-)
# 随便挑选两个维度（花萼长度，花瓣长度），使用散点图
plt.figure ( figsize=(10 , 10) )  # 画布大小（英寸）
# 绘制训练集数据
plt.scatter ( x=t0[ "Sepal.Length" ][ :int ( len ( t0 ) * 0.8 ) ] ,
              y=t0[ "Petal.Length" ][ :int ( len ( t0 ) * 0.8 ) ] ,
              color='r' , label="versicolor" )
plt.scatter ( x=t1[ "Sepal.Length" ][ :int ( len ( t1 ) * 0.8 ) ] ,
              y=t1[ "Petal.Length" ][ :int ( len ( t1 ) * 0.8 ) ] ,
              color='g' , label="setosa" )
plt.scatter ( x=t2[ "Sepal.Length" ][ :int ( len ( t2 ) * 0.8 ) ] ,
              y=t2[ "Petal.Length" ][ :int ( len ( t2 ) * 0.8 ) ] ,
              color='b' , label="virginica" )
# 绘制测试集数据
right = test_X[ result == test_y ]  # 预测正确的
wrong = test_X[ result != test_y ]  # 预测错误的
plt.scatter ( x=right[ "Sepal.Length" ] , y=right[ [ "Petal.Length" ] ] , color='c' , marker='x' , label="right" )
plt.scatter ( x=wrong[ "Sepal.Length" ] , y=wrong[ [ "Petal.Length" ] ] , color='m' , marker='>' , label="wrong" )
plt.xlabel ( "花萼长度" )
plt.ylabel ( "花瓣长度" )
plt.title ( "KNN分类结果显示" )
plt.legend ( loc="best" )  # 图例
plt.show ()

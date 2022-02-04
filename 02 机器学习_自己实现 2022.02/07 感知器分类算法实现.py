import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

data = pd.read_csv ( r"Iris.csv" )
data.drop ( "Id" , axis=1 , inplace=True )
data.drop_duplicates ( inplace=True )
data[ "Species" ] = data[ "Species" ].map ( {"versicolor": 0 , "setosa": -1 , "virginica": 1} )  # 感知器预测的结果是1与-1，0不使用
data = data[ data[ "Species" ] != 0 ]

t1 = data[ data[ "Species" ] == 1 ]
t2 = data[ data[ "Species" ] == -1 ]
t1 = t1.sample ( len ( t1 ) )
t2 = t2.sample ( len ( t2 ) )
train_X = pd.concat (
    [ t1.iloc[ :int ( len ( t1 ) * 0.8 ) , :-1 ] , t2.iloc[ :int ( len ( t2 ) * 0.8 ) , :-1 ] ] , axis=0 )
train_y = pd.concat (
    [ t1.iloc[ :int ( len ( t1 ) * 0.8 ) , -1 ] , t2.iloc[ :int ( len ( t2 ) * 0.8 ) , -1 ] ] , axis=0 )
test_X = pd.concat (
    [ t1.iloc[ int ( len ( t1 ) * 0.8 ): , :-1 ] , t2.iloc[ int ( len ( t2 ) * 0.8 ): , :-1 ] ] , axis=0 )
test_y = pd.concat (
    [ t1.iloc[ int ( len ( t1 ) * 0.8 ): , -1 ] , t2.iloc[ int ( len ( t2 ) * 0.8 ): , -1 ] ] , axis=0 )


class TrasondPerception:
    """
    感知器实现二分类
    """

    def __init__(self , alpha , times):
        """
        :param alpha: float，学习率，用来控制步长（权重调整的幅度）
        :param times: int，循环迭代的次数
        """
        self.alpha = alpha
        self.times = times
        self.w_ = None
        self.loss_ = None

    def step(self , z):
        """
        阶跃函数，作用类似于sigmoid，z >= 0则返回1，否则返回-1
        :param z:数组类型，函数的参数
        :return: int，-1或1，实现二分类
        """
        #        return 1 if z >= 0 else 0  # 标量就这么干
        return np.where ( z >= 0 , 1 , -1 )

    def fit(self , X , y):
        """
        训练
        :param X: 类数组类型，形状为[样本数量,特征数量]，待训练的样本特征（属性）
        :param y: 类数组类型，形状为[样本数量]，每个样本的目标值（分类）
        """
        X = np.asarray ( X )
        y = np.asarray ( y )
        self.w_ = np.zeros ( 1 + X.shape[ 1 ] )
        self.loss_ = [ ]
        for i in range ( self.times ):
            """
            感知器与逻辑回归的区别：
            逻辑回归中，使用所有样本计算梯度，更新权重
            感知器中，使用单个样本依次进行计算梯度，更新权重
            """
            loss = 0  # 每个样本产生的损失值，损失函数：sum（不一样=1，一样=0）
            for x , target in zip ( X , y ):
                y_hat = self.step ( np.dot ( x , self.w_[ 1: ] ) + self.w_[ 0 ] )  # 对每个样本计算预测值
                loss += y_hat != target
                # 更新权重 wj = wj + alpha * (y - y_hat) * xj
                self.w_[ 0 ] += self.alpha * (target - y_hat)
                self.w_[ 1: ] += self.alpha * (target - y_hat) * x
            self.loss_.append ( loss )  # 累积的误差值加入误差列表

    def predict(self , X):
        """
        根据参数传递的样本，对样本数据预测分类
        :param X: 类数组类型，形状为[样本数量,特征数量]，待预测的样本特征（属性）
        :return: 数组类型，预测的分类值1或-1
        """
        return self.step ( np.dot ( X , self.w_[ 1: ] ) + self.w_[ 0 ] )


p = TrasondPerception ( 0.1 , 10 )
p.fit ( train_X , train_y )
result = p.predict ( test_X )

print ( "预测结果：\n" , result == test_y )
print ( "\n预测准确率：\n" , np.sum ( result == test_y ) / len ( result ) )
print ( "查看权重：\n" , p.w_ )
print ( "查看损失：\n" , p.loss_ )

mpl.rcParams[ "font.family" ] = "SimHei"
mpl.rcParams[ "axes.unicode_minus" ] = False
plt.figure ( figsize=(10 , 10) )
plt.plot ( test_y.values , "go" , ms=15 , label="真实值" )
plt.plot ( result , "rx" , ms=15 , label="预测值" )
plt.title ( "感知器二分类" )
plt.xlabel ( "样本序号" )
plt.ylabel ( "类别" )
plt.legend ()
plt.show ()
# 绘制损失值
plt.figure ( figsize=(10 , 10) )
plt.title ( "损失列表" )
plt.xlabel ( "训练次数" )
plt.ylabel ( "损失函数" )
plt.plot ( range ( 1 , p.times + 1 ) , p.loss_ , "go-" )
plt.show ()

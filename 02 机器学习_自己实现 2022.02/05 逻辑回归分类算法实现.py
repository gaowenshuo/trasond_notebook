import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

data = pd.read_csv ( r"Iris.csv" )
data.drop ( "Id" , axis=1 , inplace=True )
data.drop_duplicates ( inplace=True )
data[ "Species" ] = data[ "Species" ].map ( {"versicolor": 0 , "setosa": 1 , "virginica": 2} )
# 由于逻辑回归是二分法，所以这里只选取0与1类别
data = data[ data[ "Species" ] != 2 ]

t0 = data[ data[ "Species" ] == 0 ]
t1 = data[ data[ "Species" ] == 1 ]
t0 = t0.sample ( len ( t0 ) )
t1 = t1.sample ( len ( t1 ) )
train_X = pd.concat (
    [ t0.iloc[ :int ( len ( t0 ) * 0.8 ) , :-1 ] , t1.iloc[ :int ( len ( t1 ) * 0.8 ) , :-1 ] ] , axis=0 )
train_y = pd.concat (
    [ t0.iloc[ :int ( len ( t0 ) * 0.8 ) , -1 ] , t1.iloc[ :int ( len ( t1 ) * 0.8 ) , -1 ] ] , axis=0 )
test_X = pd.concat (
    [ t0.iloc[ int ( len ( t0 ) * 0.8 ): , :-1 ] , t1.iloc[ int ( len ( t1 ) * 0.8 ): , :-1 ] ] , axis=0 )
test_y = pd.concat (
    [ t0.iloc[ int ( len ( t0 ) * 0.8 ): , -1 ] , t1.iloc[ int ( len ( t1 ) * 0.8 ): , -1 ] ] , axis=0 )


class TrasondLogisticRegression:
    """
    逻辑回归：
    先梯度下降线性回归，再带入sigmoid函数进行二分类
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

    def sigmoid(self , z):
        """
        将R上的连续值转化到（0，1）概率区间的函数
        :param z: float，自变量，z = w.T * X
        :return: s(z)：float，(0,1)区间，样本属于类别1的概率值，用来作为结果的预测
        当s >= 0.5即z >= 0时，判定为类别1，否则判定为类别0
        """
        return 1.0 / (1.0 + np.exp ( -z ))

    def fit(self , X , y):
        """
        训练
        :param X: 类数组类型，形状为[样本数量,特征数量]，待训练的样本特征（属性）
        :param y: 类数组类型，形状为[样本数量]，每个样本的目标值（标签）
        """

        X = np.asarray ( X )
        y = np.asarray ( y )
        self.w_ = np.zeros ( 1 + X.shape[ 1 ] )
        self.loss_ = [ ]
        for i in range ( self.times ):
            z = np.dot ( X , self.w_[ 1: ] ) + self.w_[ 0 ]
            p = self.sigmoid ( z )  # 计算概率值
            # 逻辑回归的损失函数/目标函数/代价函数：J（w） = -sum（yi*log s（zi）+（1-yi）*log （1 - s（zi））），其中真实值yi=0时只有后面那一项，s（zi）越接近0函数值越小，yi=1时只有前面那一项，s（zi）越接近1函数值越小
            cost = -np.sum ( y * np.log ( p ) ) + (1 - y) * np.log ( 1 - p )  # 计算损失值
            self.loss_.append ( cost )
            # 调整权重值，根据公式调整为Wj = Wj + alpha * sum((y - s(z)) * Xj) Wj是向量W的第j项，Xj是矩阵X的第j列
            self.w_[ 0 ] += self.alpha * np.sum ( y - p )
            self.w_[ 1: ] += self.alpha * np.dot ( X.T , y - p )
        return None

    def predict_proba(self , X):
        """
        根据参数传递的样本，对样本数据预测概率值
        :param X: 类数组类型，形状为[样本数量,特征数量]，待预测的样本特征（属性）
        :return: 数组类型，预测的概率值
        """
        X = np.asarray ( X )
        z = np.dot ( X , self.w_[ 1: ] ) + self.w_[ 0 ]
        p = self.sigmoid ( z )  # 分类为1的概率
        p = p.reshape ( -1 , 1 )
        # 横向拼接为（为0的概率，为1的概率）的形式
        return np.concatenate ( [ 1 - p , p ] , axis=1 )

    def predict(self , X):
        """
        根据参数传递的样本，对样本数据进行预测
        :param X: 类数组类型，形状为[样本数量,特征数量]，待预测的样本特征（属性）
        :return: 数组类型，预测的分类值
        """
        return np.argmax ( self.predict_proba ( X ) , axis=1 )


# 鸢尾花特征列都在同一个数量级，可以不用标准化
lr = TrasondLogisticRegression ( alpha=0.01 , times=20 )
lr.fit ( train_X , train_y )
result = lr.predict ( test_X )
print ( "预测结果：\n" , result == test_y )
print ( "\n预测准确率：\n" , np.sum ( result == test_y ) / len ( result ) )

mpl.rcParams[ "font.family" ] = "SimHei"
mpl.rcParams[ "axes.unicode_minus" ] = False
plt.figure ( figsize=(10 , 10) )
plt.plot ( result , "ro" , ms=15 , label="预测值" )
plt.plot ( test_y.values , "go" , label="真实值" )
plt.title("逻辑回归")
plt.xlabel("样本序号")
plt.ylabel("类别")
plt.legend()
plt.show()
# 绘制损失值
plt.figure ( figsize=(10 , 10) )
plt.title ( "损失列表" )
plt.xlabel ( "训练次数" )
plt.ylabel ( "损失函数" )
plt.plot ( range ( 1 , lr.times + 1 ) , lr.loss_ , "go-" )
plt.show ()
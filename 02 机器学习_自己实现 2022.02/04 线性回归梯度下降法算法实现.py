import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

data = pd.read_csv ( r"Boston.csv" )
data = data.drop ( "Id" , axis=1 )


# 梯度下降法中，不同特征值间数量级差距大，需要进行标准化
class TrasondStandardScaler:
    """
    对数据标准化处理
    """

    def __init__(self):
        self.std_ = 0
        self.mean_ = 0

    def fit(self , X):
        """
        根据传递的样本，计算每个特征列的均值与标准差
        :param X: 类数组类型，训练数据，用来计算均值与标准差
        """
        X = np.asarray ( X )
        self.std_ = np.std ( X , axis=0 )  # 计算每一列的标准差
        self.mean_ = np.mean ( X , axis=0 )  # 计算每一列的均值
        return None

    def transform(self , X):
        """
        对给定的X进行标准化处理（将X的每一列变成标准正态分布的数据）
        :param X: 类数组类型，待转换的数据
        :return: 类数组类型，X转换成标准正态分布后的结果
        """
        return (X - self.mean_) / self.std_

    def fit_transform(self , X):
        """
        对数据进行训练并转换，返回转换之后的结果
        :param X: 类数组类型，待转换的数据
        :return: 类数组类型，X转换成标准正态分布后的结果
        """
        self.fit ( X )
        return self.transform ( X )


t = data.sample ( len ( data ) )
train_X = t.iloc[ :int ( len ( t ) * 0.8 ) , :-1 ]
train_y = t.iloc[ :int ( len ( t ) * 0.8 ) , -1 ]
test_X = t.iloc[ int ( len ( t ) * 0.8 ): , :-1 ]
test_y = t.iloc[ int ( len ( t ) * 0.8 ): , -1 ]
# 为了避免每个特征数量级的不同，从而对梯度下降带来影响，所以对每个特征进行标准化
s = TrasondStandardScaler ()
train_X = s.fit_transform ( train_X )
test_X = s.transform ( test_X )  # 保持之前的参数，不重新fit了
s2 = TrasondStandardScaler ()
train_y = s2.fit_transform ( train_y )
test_y = s2.transform ( test_y )


class TrasondLinearRegression2:
    """
    线性回归（梯度下降法）
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

    def fit(self , X , y):
        """
        训练
        :param X: 类数组类型，形状为[样本数量,特征数量]，待训练的样本特征（属性）
        :param y: 类数组类型，形状为[样本数量]，每个样本的目标值（标签）
        """
        X = np.asarray ( X )
        y = np.asarray ( y )
        # 创建权重的向量，初始值随意，长度比特征数量多1（多的是截距）
        self.w_ = np.zeros ( 1 + X.shape[ 1 ] )
        # 创建损失列表，用来保存每次迭代后的损失值，损失计算（损失函数/目标函数）：（预测值-真实值）的平方和 除以2
        self.loss_ = [ ]
        # 进行循环，多次迭代，在每次迭代过程中不断调整权重值，使损失值不断减小
        for i in range ( self.times ):
            y_hat = np.dot ( X , self.w_[ 1: ] ) + self.w_[ 0 ]  # 计算预测值，矩阵乘法，注意w的截距
            error = y - y_hat  # 真实值与预测值的差距
            self.loss_.append ( np.sum ( error ** 2 ) / 2 )  # 损失值加入损失列表
            # 根据差距调整权重，梯度下降法公式：Wj = Wj + alpha * sum((y - y_hat) * Xj) Wj是向量W的第j项，Xj是矩阵X的第j列
            self.w_[ 0 ] += self.alpha * np.sum ( error )  # 对于截距W0来说，X0都是1
            self.w_[ 1: ] += self.alpha * np.dot ( X.T , error )  # 注意是X的某一列和y-y_hat，对应位置相乘求和
        return None

    def predict(self , X):
        """
        根据参数传递的样本，对样本数据进行预测
        :param X: 类数组类型，形状为[样本数量,特征数量]，待预测的样本特征（属性）
        :return: 数组类型，预测的结果
        """
        X = np.asarray ( X )
        result = np.dot ( X , self.w_[ 1: ] ) + self.w_[ 0 ]
        return result


lr = TrasondLinearRegression2 ( alpha=0.0005 , times=50 )
lr.fit ( train_X , train_y )
result = lr.predict ( test_X )
print ( "预测方差：\n" , np.mean ( (result - test_y) ** 2 ) )
print ( "查看权重：\n" , lr.w_ )
print ( "查看损失：\n" , lr.loss_ )

mpl.rcParams[ "font.family" ] = "SimHei"
mpl.rcParams[ "axes.unicode_minus" ] = False
plt.figure ( figsize=(10 , 10) )
plt.plot ( result , "ro-" , label="预测值" )
plt.plot ( test_y.values , "go--" , label="真实值" )
plt.title ( "线性回归预测-梯度下降法" )
plt.xlabel ( "样本序号" )
plt.ylabel ( "房价" )
plt.legend ()
plt.show ()
# 绘制累计误差值
plt.figure ( figsize=(10 , 10) )
plt.title ( "损失列表" )
plt.xlabel ( "训练次数" )
plt.ylabel ( "损失函数" )
plt.plot ( range ( 1 , lr.times + 1 ) , lr.loss_ , "go-" )
plt.show ()

# 直线拟合可视化
# 因为房价分析涉及多个维度，不方便进行可视化，选取一个维度（RM），画出直线实现拟合
train_X = t.iloc[ :int ( len ( t ) * 0.8 ) , 5:6 ]  # 只截取一列，为了防止变成一维Series结构，写成切片
train_y = t.iloc[ :int ( len ( t ) * 0.8 ) , -1 ]
test_X = t.iloc[ int ( len ( t ) * 0.8 ): , 5:6 ]
test_y = t.iloc[ int ( len ( t ) * 0.8 ): , -1 ]
s = TrasondStandardScaler ()
train_X = s.fit_transform ( train_X )
test_X = s.transform ( test_X )
s2 = TrasondStandardScaler ()
train_y = s2.fit_transform ( train_y )
test_y = s2.transform ( test_y )
lr = TrasondLinearRegression2 ( alpha=0.0005 , times=50 )
lr.fit ( train_X , train_y )
result = lr.predict ( test_X )
plt.figure ( figsize=(10 , 10) )
plt.scatter ( train_X[ "rm" ] , train_y )
# 直线方程：y = W0 + W1 * x
x = np.arange ( -5 , 5 , 0.1 )
y = lr.w_[ 0 ] + lr.w_[ 1 ] * x
plt.plot ( x , y , "r" )
# 等价于： plt.plot ( x , lr.predict ( x.reshape ( -1 , 1 ) ) , "r" )
plt.title ( "直线拟合" )
plt.xlabel ( "RM" )
plt.ylabel ( "MEDV" )
plt.show ()

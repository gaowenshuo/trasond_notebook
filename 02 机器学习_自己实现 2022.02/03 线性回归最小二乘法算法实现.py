import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

data = pd.read_csv ( r"Boston.csv" )
"""
字段说明：
CRIM：房屋所在镇的犯罪率
ZN：面积大于25000平方英尺住宅所占的比例
INDUS：房屋所在镇非零售区域所占的比例
CHAS：房屋是否位于河边，如果位于河边为1，否则为0
NOX：一氧化氮浓度
RM：平均房间数量
AGE：1940年前建成房屋所占的比例
DIS：房屋距离波士顿五大就业中心的加权距离
RAD：距离房屋最近的公路
TAX：财产税收额度
PTRATIO：房屋所在镇师生比例
B：1000*（房屋所在镇非美籍人口所占的比例-0.63）**2
LSTAT：弱势群体人口所占的比例
MEDV：房屋的平均价格
"""
data = data.drop ( "Id" , axis=1 )
# 查看基本信息和各个特征列是否有缺失值
data.info ()
# 是否有重复值
# data.duplicated ().any ()

t = data.sample ( len ( data ) )
# 考虑到截距，需要给X前面加上一列1
new_columns = t.columns.insert ( 0 , "intercept" )
t = t.reindex ( columns=new_columns , fill_value=1 )

train_X = t.iloc[ :int ( len ( t ) * 0.8 ) , :-1 ]
train_y = t.iloc[ :int ( len ( t ) * 0.8 ) , -1 ]
test_X = t.iloc[ int ( len ( t ) * 0.8 ): , :-1 ]
test_y = t.iloc[ int ( len ( t ) * 0.8 ): , -1 ]


class TrasondLinearRegression:
    """
    线性回归（最小二乘法）
    """

    def __init__(self):
        self.w_ = None

    def fit(self , X , y):
        """
        根据提供的训练数据，对模型进行训练
        :param X: 类数组类型，形状为[样本数量,特征数量]，待训练的样本特征（属性）
        :param y: 类数组类型，形状为[样本数量]，每个样本的目标值（标签）
        """
        # 线性回归：y = WT * X = W0（截距） + W1X1 + W2X2 +...，找到向量W
        # 最小二乘法正规方程公式：W = (XT * X)-1 * XT * y
        X = np.asmatrix ( X.copy () )  # 拷贝是因为函数必须接收完整的数组而不是数组对象的一部分（例如由其他对象切片而来），否则无法完成转换
        y = np.asmatrix ( y ).reshape ( -1 , 1 )  # 一维（即向量）不存在上述问题不用拷贝，矩阵运算时需要转换后转化为二维（列向量的形状）（-1是自动的意思）
        self.w_ = (X.T * X).I * X.T * y  # 注意ndarray和矩阵的乘法规则不同
        return None

    def predict(self , X):
        """
        根据参数传递的样本，对样本数据进行预测
        :param X: 类数组类型，形状为[样本数量,特征数量]，待预测的样本特征（属性）
        :return: 数组类型，预测的结果
        """
        X = np.asmatrix ( X.copy () )
        result = X * self.w_  # 通过矩阵乘法计算预测值
        return np.asarray ( result ).ravel ()  # 扁平化处理，转换为一维ndarray


lr = TrasondLinearRegression ()
lr.fit ( train_X , train_y )
result = lr.predict ( test_X )
# print ( "预测结果：\n" , result )
# print ( "正确答案：\n" , test_y.values )
print ( "预测方差：\n" , np.mean ( (result - test_y) ** 2 ) )
print ( "查看权重：\n" , lr.w_ )

mpl.rcParams[ "font.family" ] = "SimHei"
mpl.rcParams[ "axes.unicode_minus" ] = False
plt.figure ( figsize=(10 , 10) )
plt.plot ( result , "ro-" , label="预测值" )
plt.plot ( test_y.values , "go--" , label="真实值" )
plt.title ( "线性回归展示-最小二乘法" )
plt.xlabel ( "样本序号" )
plt.ylabel ( "房价" )
plt.legend ()
plt.show ()

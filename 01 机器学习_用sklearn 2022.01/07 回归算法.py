import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score


def demo_linear_regression():
    """
    线性回归是指用回归方程对特征值和目标值关系进行建模
    损失方程：训练样本和预测函数之差的平方的和，即最小二乘法
    优化算法：减少损失：正规方程（计算量大一步到位）/梯度下降（选择学习率迭代求解）
    正规方程：
    sklearn.linear_model.LinearRegression(fit_intercept=True)（是否计算偏置）
    LinearRegression.coef_:回归系数
    LinearRegression.intercept_:偏置
    随机梯度下降：
    sklearn.linear_model.SGDRegressor(loss="squared_loss",fit_intercept=True,learning_rate='invscaling',eta0=0.01)（损失类型，是否计算偏置，学习率填充如'constant','optimal',
    'invscaling'用不同方法从eta0计算eta）
    SGDRegression.coef_:回归系数
    SGDRegression.intercept_:偏置

    回归性能评估：均方误差MSE
    sklearn.metrics.mean_squared_error(y_true,y_pred)
    """
    # 本来用load_boston现在貌似用不了了
    house = fetch_california_housing ()
    x_train , x_test , y_train , y_test = train_test_split ( house.data , house.target , test_size=0.2 )
    #    transfer = StandardScaler ()
    #    x_train = transfer.fit_transform ( x_train )
    #    x_test = transfer.transform ( x_test )
    estimator = LinearRegression ()
    estimator.fit ( x_train , y_train )
    y_predict = estimator.predict ( x_test )
    print ( "线性回归预测房价，正规方程权重参数为：\n" , estimator.coef_ )
    print ( "线性回归预测房价，正规方程均方误差为：\n" , mean_squared_error ( y_test , y_predict ) )
    estimator = SGDRegressor ()
    estimator.fit ( x_train , y_train )
    y_predict = estimator.predict ( x_test )
    print ( "线性回归预测房价，随机梯度下降均方误差为：\n" , mean_squared_error ( y_test , y_predict ) )
    """
    欠拟合：训练数据上不能获得更好的拟合，且测试数据不能很好的拟合数据（模型过于简单，解决方法：增加特征数量）
    过拟合：训练数据上能获得更好的拟合，但测试数据不能很好的拟合数据（模型过于复杂，解决办法：L2/L1正则化）
    岭回归：加入L2正则化的线性回归，随机梯度下降
    sklearn.linear_model.Ridge(alpha=1.0,fit_intercept=True,solver="auto",normalize=False)（正则化力度，优化方法如"sag"，是否标准化）
    Ridge.coef_:回归系数
    Ridge.intercept_:偏置
    还有个RidgeCV
    """
    estimator = Ridge ()
    estimator.fit ( x_train , y_train )
    y_predict = estimator.predict ( x_test )
    print ( "岭回归预测房价均方误差为：\n" , mean_squared_error ( y_test , y_predict ) )
    return None


def demo_logistic_regression():
    """
    逻辑回归实际上是一种二分类算法
    将一个线性回归结果输入到sigmoid函数（激活函数）得到一个[0，1]间概率值，默认0.5为阈值，超过0.5判断为正例1，否则判断为反例0，默认样本量少的为正例
    sklearn.linear_model.LogisticRegression(solver='liblinear',penalty='l2',C=1.0)（优化求解方式如'sag'，正则化种类，正则化力度）
    """
    column_name = [ 'Sample code number' , 'Clump Thickness' , 'Uniformity of Cell Size' , 'Uniformity of Cell Shape' ,
                    'Marginal Adhesion' , 'Single Epithelial Cell Size' , 'Bare Nuclei' , 'Bland Chromatin' ,
                    'Normal Nucleoli' , 'Mitoses' , 'Class' ]
    data = pd.read_csv (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data" ,
        names=column_name )
    data = data.replace ( to_replace='?' , value=np.nan )
    data = data.dropna ()
    x = data[ column_name[ 1:10 ] ]
    y = data[ column_name[ 10 ] ]
    x_train , x_test , y_train , y_test = train_test_split ( x , y , test_size=0.3 )
    transfer = StandardScaler ()
    x_train = transfer.fit_transform ( x_train )
    x_test = transfer.transform ( x_test )
    estimator = LogisticRegression ()
    estimator.fit ( x_train , y_train )
    print ( "逻辑回归预测癌症类别，得出来的权重：\n" , estimator.coef_ )
    print ( "逻辑回归预测癌症类别，预测的准确率：\n" , estimator.score ( x_test , y_test ) )
    """
    这里由于两类的样本量差距大，评价的指标就不仅仅是准确率了（例如大部分全是某一类，就算模型把所有都预测为这一类，准确率也是很高的，另一类的准确率就很重要）
    精确率：预测结果正例中真的是正例（真阳性占真阳性加假阳性）
    召回率：真实的整理中预测结果为正例（真阳性占真阳性加假阴性）
    sklearn.metrics.classification_report(y_true,y_pred,labels=[],target_names)
    """
    print ( "精确率和召回率：" , classification_report ( y_test , estimator.predict ( x_test ) , labels=[ 2 , 4 ] ,
                                                 target_names=[ '良性' , '恶性' ] ) )
    """
    AUC是ROC曲线的面积，指随机取一对正负样本，正样本得分大于负样本的概率，0.5-1，越接近1越好
    sklearn.metrics.roc_auc_score(y_test, y_score)
    """
    y_test = np.where ( y_test > 2.5 , 1 , 0 )
    print ( "AUC指标：" , roc_auc_score ( y_test , estimator.predict ( x_test ) ) )
    return None


if __name__ == "__main__":
    demo_linear_regression ()
    demo_logistic_regression ()

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


def demo_transformer_and_estimator():
    """
    分类：目标是类别
    回归：目标是数
    聚类：无监督学习

    转换器：特征工程的数据转换的父类
    实例化 -》 fit() -》 transform()
    以StandardScaler为例
    fit_transform()是由fit()（计算每一列平均值/标准差，参数存到转换器里）和transform()（代入公式进行最后的转换）合成

    估计器：机器学习算法的父类
    实例化 -》 fit(x_train, y_train)（生成模型）
    -》 score(x_test, y_test)（结果精度） 或 y_predict = predict(x_test)（预测结果）

    下面是分类算法的各个算法，即各个估计器
    """
    return None


def demo_neighbors():
    """
    第一种：k-近邻算法（KNN）
    原理：找到距离最短的推断出类别，如果特征空间中k个最相似的大多数属于某一类，则样本属于这一类
    欧式距离：对应位置之差的平方的和开方
    sklearn.neighbors.KNeighborsClassifier(n_neighbors=5,algorithm='auto')（邻居数，算法包括'ball_tree','kd_tree','brute'）
    """
    iris = load_iris ()
    x_train , x_test , y_train , y_test = train_test_split ( iris.data , iris.target , test_size=0.2 )
    transfer = StandardScaler ()
    x_train = transfer.fit_transform ( x_train )
    x_test = transfer.transform ( x_test )  # 注意这句，不重新fit了，两个集需要用同样的处理
    estimator = KNeighborsClassifier ( n_neighbors=10 )  # 理论上k = 根号（样本数），过小：异常值影响，过大：样本不均衡影响
    estimator.fit ( x_train , y_train )
    y_predict = estimator.predict ( x_test )
    print ( "KNN预测iris的结果为：\n" , y_predict == y_test )
    print ( "KNN预测iris准确率为：\n" , estimator.score ( x_test , y_test ) )
    return None


def demo_grid_search():
    """
    交叉验证：cross validation：数据分为几个部分，依次作为验证集（其他为训练集），多次训练模型求平均准确率
    超参数搜索调优：模型中手动指定的参数例如上述的k值，通过交叉验证选出最优参数组合
    sklearn.model_selection.GridSearchCV(estimator,param_grid,cv)（估计器对象，参数列表dict，几折交叉验证）
    fit:输入训练数据
    score:准确率
    结果：
    GridSearchCV.best_score_:交叉验证最好结果
    GridSearchCV.best_estimator_:最好的参数模型
    GridSearchCV.best_params_:最好的参数
    GridSearchCV.cv_results_:每次交叉验证的验证集准确率结果和训练集准确率结果
    """
    iris = load_iris ()
    x_train , x_test , y_train , y_test = train_test_split ( iris.data , iris.target , test_size=0.2 )
    transfer = StandardScaler ()
    x_train = transfer.fit_transform ( x_train )
    x_test = transfer.transform ( x_test )
    estimator = KNeighborsClassifier ()
    param = {"n_neighbors": [ 1 , 3 , 5 , 7 , 9 , 11 ]}
    estimator = GridSearchCV ( estimator , param_grid=param , cv=10 )
    estimator.fit ( x_train , y_train )

    print ( "最佳参数：\n" , estimator.best_params_ )
    print ( "最佳结果：\n" , estimator.best_score_ )
    print ( "最佳估计器：\n" , estimator.best_estimator_ )
    print ( "交叉验证结果：\n" , estimator.cv_results_ )
    return None


if __name__ == "__main__":
    demo_transformer_and_estimator ()
    demo_neighbors ()
    demo_grid_search ()

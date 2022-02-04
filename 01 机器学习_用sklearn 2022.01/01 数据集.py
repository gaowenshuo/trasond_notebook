from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def demo_datasets():
    """
    sklearn:分类/聚类/回归，特征工程，模型选择/调优
    sklearn数据集（sklearn.datasets）:
    小数据集：datasets.load_*()
    大数据集：datasets.fetch_*(data_home="~/scikit_learn_data/")（下载位置）
    得到的数据类型：datasets.base.Bunch（字典格式）
    键或属性调用：
    data:特征数据数组，numpy.ndarray二维数组
    target:标签数组，numpy.ndarray一维数组
    DESCR:数据描述
    feature_names:特征名
    target_names:标签名
    """
    iris = load_iris ()
    # print(iris)
    print ( "特征值：\n" , iris.data )
    print ( "n_samples, n_features：\n" , iris.data.shape )
    print ( "目标值：\n" , iris.target )
    print ( "特征名：\n" , iris.feature_names )
    print ( "目标值名：\n" , iris.target_names )
    print ( "鸢尾花的描述：\n" , iris.DESCR )
    return None


def demo_split():
    """
    数据集的划分：
    训练集：70%，80%，75%
    测试集：30%，20%，25%
    sklearn.model_selection.train_test_split(arrays,*options)
    x:数据集的特征值
    y:数据集的标签值
    test_size:测试集大小，float
    random_state:随机数种子
    return:训练集特征值，测试集特征值，训练标签，测试标签
    """
    iris = load_iris ()
    # 分别是训练集特征值，测试集特征值，训练集目标值，测试集目标值
    x_train , x_test , y_train , y_test = train_test_split ( iris.data , iris.target , test_size=0.2 )
    print ( "x_train的大小：\n" , x_train.shape )
    return None


if __name__ == "__main__":
    demo_datasets ()
    demo_split ()

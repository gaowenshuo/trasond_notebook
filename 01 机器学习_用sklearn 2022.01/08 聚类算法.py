import joblib
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_california_housing , load_iris
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score


def demo_save_and_load():
    """
    保存和加载模型：
    sklearn.externals.joblib
    保存：joblib.dump(estimator, 'test.pkl')
    加载：estimator = joblib.load('test.pkl')
    """
    house = fetch_california_housing ()
    x_train , x_test , y_train , y_test = train_test_split ( house.data , house.target , test_size=0.2 )
    #    transfer = StandardScaler ()
    #    x_train = transfer.fit_transform ( x_train )
    #    x_test = transfer.transform ( x_test )
    estimator = LinearRegression ()
    estimator.fit ( x_train , y_train )
    joblib.dump ( estimator , "test.pkl" )
    estimator = joblib.load ( "test.pkl" )
    y_predict = estimator.predict ( x_test )
    print ( "用加载的模型预测房价均方误差为：\n" , mean_squared_error ( y_test , y_predict ) )
    return None


def demo_k_means():
    """
    K-means聚类原理：
    随机设置特征空间内K个点作为初始聚类中心，每个点计算到每个中心的距离，选择最近的作为标记类别，然后重新计算出平均值作为每个聚类的新中心点，重复直到中心点不变
    sklearn.cluster.KMeans(n_clusters=8,init='k-means++')（聚类数量，初始化方法）
    labels_:默认标记类型，可以和真实值比较（不是值比较）
    评价指标：轮廓系数，[-1，1]区间，越接近1越好，代表内聚度和分离度都相对较优
    这个值由“外部距离（到其他族群平均距离）”和“内部距离（到同一族群其他点平均距离）”决定，外部距离越大，内部距离越小，轮廓系数越大
    sklearn.metrics.silhouette_score(X,labels)（特征值，被聚类标记的目标值）
    """
    iris = load_iris ()
    data = iris.data
    estimator = KMeans ( n_clusters=3 )
    estimator.fit ( data )
    pre = estimator.predict ( data )
    print ( "对鸢尾花聚类结果：\n" , pre )
    print ( "边缘系数：\n" , silhouette_score ( data , pre ) )
    return None


if __name__ == "__main__":
    demo_save_and_load ()
    demo_k_means ()

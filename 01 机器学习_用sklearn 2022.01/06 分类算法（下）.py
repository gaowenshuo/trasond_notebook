from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def demo_naive_bayes():
    """
    第二种：朴素贝叶斯算法
    原理：假设样本属性独立，计算条件概率和联合概率
    sklearn.naive_bayes.MultinomialNB(alpha=1.0)（拉普拉斯平滑系数）
    """
    news = fetch_20newsgroups ( subset="all" )
    x_train , x_test , y_train , y_test = train_test_split ( news.data , news.target , test_size=0.2 )
    transfer = TfidfVectorizer ()
    x_train = transfer.fit_transform ( x_train )
    x_test = transfer.transform ( x_test )
    estimator = MultinomialNB ()
    estimator.fit ( x_train , y_train )
    y_predict = estimator.predict ( x_test )
    print ( "朴素贝叶斯预测新闻类别的结果为：\n" , y_predict == y_test )
    print ( "朴素贝叶斯预测新闻类别准确率为：\n" , estimator.score ( x_test , y_test ) )
    return None


def demo_tree():
    """
    第三种：决策树
    类似于分支结构
    与信息熵（H，单位为比特）有关（信息越多，猜测的不确定性越小）
    决策树的划分依据：信息增益，表示得知X的信息的不确定性减少的程度使得类Y的信息熵减少的程度
    sklearn.tree.DecisionTreeClassifier(criterion='gini',max_depth,random_state)（系数可以是'entropy'，树的深度，随机数种子）
    """
    iris = load_iris ()
    x_train , x_test , y_train , y_test = train_test_split ( iris.data , iris.target , test_size=0.2 )
    transfer = StandardScaler ()
    x_train = transfer.fit_transform ( x_train )
    x_test = transfer.transform ( x_test )
    estimator = DecisionTreeClassifier ( max_depth=5 )
    estimator.fit ( x_train , y_train )
    y_predict = estimator.predict ( x_test )
    print ( "决策树预测iris的结果为：\n" , y_predict == y_test )
    print ( "决策树预测iris准确率为：\n" , estimator.score ( x_test , y_test ) )
    """
    决策树可视化：
    sklearn.tree.export_graphviz(estimator,out_file='tree.dot',feature_names)（估计器，输出位置，特征名）
    """
    export_graphviz ( estimator , out_file="iris_tree.dot" , feature_names=iris.feature_names )
    # 可以去webgraphviz.com生成图片
    return None


def demo_random_forest():
    """
    集成学习方法：建立几个模型组合，各自独立学习预测，结合成组合预测

    随机森林：包含多个决策树的分类器，输出的类别由众数决定
    建树过程：BootStrap采样（有放回的随机抽样）
    sklearn.ensemble.RandomForestClassifier(n_estimators=10,criterion='gini',max_depth,bootstrap=True,random_state,min_samples_split)（森林里的树木数量，测量方法，数的最大深度，是否放回抽样，随机数种子，节点划分最小样本数）
    """
    iris = load_iris ()
    x_train , x_test , y_train , y_test = train_test_split ( iris.data , iris.target , test_size=0.2 )
    transfer = StandardScaler ()
    x_train = transfer.fit_transform ( x_train )
    x_test = transfer.transform ( x_test )
    estimator = RandomForestClassifier ()
    param = {"n_estimators": [ 10 , 50 , 100 , 200 , 300 ] , "max_depth": [ 2 , 3 , 5 , 8 , 10 ]}
    estimator = GridSearchCV ( estimator , param_grid=param , cv=2 )
    estimator.fit ( x_train , y_train )
    y_predict = estimator.predict ( x_test )
    print ( "随机森林预测iris的结果为：\n" , y_predict == y_test )
    print ( "随机森林预测iris准确率为：\n" , estimator.score ( x_test , y_test ) )
    return None


if __name__ == "__main__":
    demo_naive_bayes ()
    demo_tree ()
    demo_random_forest ()

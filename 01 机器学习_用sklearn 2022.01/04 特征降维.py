from pandas import DataFrame
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA


def demo_selection():
    """
    特征降维：减少特征值的列数：特征选择，主成分分析
    特征选择：选择主要的过滤冗余的
    sklearn.feature_selection.VarianceThreshold(threshold=0.0)
    删除方差低于阈值的特征，默认值是删除具有相同值的特征
    Variance.fit_transform(X)
    其他：相关系数
    from scipy.stats import pearsonr
    pearsonr(data[i], data[j])
    相关系数>0：正相关，<0：负相关，|r|越接近1，线性关系越密切，越接近0越弱
    """
    data = DataFrame ( [ [ 40920 , 8.326976 , 0.953952 , 3 ] ,
                         [ 14488 , 7.153469 , 1.673904 , 2 ] ,
                         [ 26052 , 1.441871 , 0.805124 , 1 ] ,
                         [ 75136 , 13.147394 , 0.428964 , 1 ] ,
                         [ 38344 , 1.669788 , 0.134296 , 1 ] ] ,
                       columns=[ "milage" , "Liters" , "Consumtime" , "target" ] )
    print ( data )
    transfer = VarianceThreshold ( threshold=1 )
    data_new = transfer.fit_transform ( data )
    print ( "删除低方差特征的结果：\n" , data_new )
    return None


def demo_pca():
    """
    主成分分析PCA：尽可能损失少量信息降低原数据的维数，会舍弃原有数据创造新的变量
    想象照相时选择角度保留最多信息，或将平面上几个点投影到一条线上使得区分度最高
    sklearn.decomposition.PCA(n_components)（小数：保留百分之多少的信息，整数：减少到多少特征）
    PCA.fit_transform
    """
    data = [ [ 2 , 8 , 4 , 5 ] ,
             [ 6 , 3 , 0 , 8 ] ,
             [ 5 , 4 , 9 , 1 ] ]
    transfer = PCA ( n_components=0.9 )
    data_new = transfer.fit_transform ( data )
    print ( "保留90%信息，降维结果：\n" , data_new )
    transfer = PCA ( n_components=3 )
    data_new = transfer.fit_transform ( data )
    print ( "降维到3维的结果：\n" , data_new )
    return None


if __name__ == "__main__":
    demo_selection ()
    demo_pca ()

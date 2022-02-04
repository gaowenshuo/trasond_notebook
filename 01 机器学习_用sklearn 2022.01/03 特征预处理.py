from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


def demo_minmax():
    """
    特征预处理：去除量纲
    归一化：线性压缩到指定区间
    sklearn.preprocessing.MinMaxScaler(feature_range=(0,1))（压缩到的区间）
    MinMaxScaler.fit_transform(X):返回相同形状的array
    (x - min) / (max - min)
    """
    data = DataFrame ( [ [ 40920 , 8.326976 , 0.953952 , 3 ] ,
                         [ 14488 , 7.153469 , 1.673904 , 2 ] ,
                         [ 26052 , 1.441871 , 0.805124 , 1 ] ,
                         [ 75136 , 13.147394 , 0.428964 , 1 ] ,
                         [ 38344 , 1.669788 , 0.134296 , 1 ] ] ,
                       columns=[ "milage" , "Liters" , "Consumtime" , "target" ] )
    print ( data )
    transfer = MinMaxScaler ()
    data_new = transfer.fit_transform ( data )
    print ( "最小值最大值归一化处理的结果：\n" , data_new )
    return None


def demo_stand():
    """
    标准化：压缩到均值为0，标准差为1范围内，不易受异常值影响
    sklearn.preprocessing.StandardScaler()
    (x - mean) / std
    """
    data = DataFrame ( [ [ 40920 , 8.326976 , 0.953952 , 3 ] ,
                         [ 14488 , 7.153469 , 1.673904 , 2 ] ,
                         [ 26052 , 1.441871 , 0.805124 , 1 ] ,
                         [ 75136 , 13.147394 , 0.428964 , 1 ] ,
                         [ 38344 , 1.669788 , 0.134296 , 1 ] ] ,
                       columns=[ "milage" , "Liters" , "Consumtime" , "target" ] )
    transfer = StandardScaler ()
    data_new = transfer.fit_transform ( data )
    print ( "标准化的结果：\n" , data_new )
    print ( "每一列的平均值：\n" , transfer.mean_ )
    print ( "每一列的方差：\n" , transfer.var_ )
    return None


if __name__ == "__main__":
    demo_minmax ()
    demo_stand ()

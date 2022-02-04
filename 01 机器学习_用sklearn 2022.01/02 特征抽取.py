from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


def demo_dict_feature_extraction():
    """
    特征工程三步：特征抽取，特征预处理，特征降维
    特征抽取：字典/文本/图像
    字典特征抽取：
    sklearn.feature_extraction.DictVectorizer(sparse=True,...)（是否返回sparse矩阵）
    DictVectorizer.fit_transform(X):X为字典或包含字典的迭代器，返回特征值化（one-hot编码）后的矩阵
    DictVectorizer.inversion_transform(X):X为array数组或sparse矩阵，返回转换之前的数据格式
    DictVectorizer.get_feature_names():返回类别名称
    注：sparse矩阵（稀疏矩阵）：存储非0值的坐标
    """
    data = [ {'city': '北京' , 'temperature': 100} ,
             {'city': '上海' , 'temperature': 60} ,
             {'city': '重庆' , 'temperature': 10} ,
             {'city': '深圳' , 'temperature': 30} ]
    transfer = DictVectorizer ()
    data_new = transfer.fit_transform ( data )
    print ( "返回的sparse结果：\n" , data_new )
    transfer = DictVectorizer ( sparse=False )
    data_new = transfer.fit_transform ( data )
    print ( "返回的array结果：\n" , data_new )
    print ( "特征名字：\n" , transfer.get_feature_names_out () )
    return None


def demo_text_feature_extraction():
    """
    文本特征抽取
    第一种是计数：
    sklearn.feature_extraction.text.CountVectorizer(stop_words=[])（停用词）
    CountVectorizer.fit_transform(X):X为文本或包含文本字符串的可迭代对象，返回sparse矩阵
    CountVectorizer.inverse_transform(X)
    CountVectorizer.get_feature_names():返回单词列表
    第二种是用TF-IDF指标：
    sklearn.feature_extraction.text.TfidfVectorizer
    """
    data = [ "life is short, i like python" , "life is too too long, i dislike python" ]
    transfer = CountVectorizer ( stop_words=[ "is" ] )
    data_new = transfer.fit_transform ( data )
    print ( "特征抽取结果：\n" , data_new.toarray () )  # 将sparse转为array
    print ( "返回特征名字：\n" , transfer.get_feature_names_out () )
    """
    Tf: Term frequency 词频：某一个给定的词语在该文件中出现的频率
    Idf: Inverse document frequency 逆向文档频率：总文件数目除以包含该词语文件的数目，再取以10为底的对数
    Tf * Idf: 重要程度
    """
    transfer = TfidfVectorizer ( stop_words=[ "is" ] )
    data_new = transfer.fit_transform ( data )
    print ( "特征抽取结果：\n" , data_new.toarray () )  # 将sparse转为array
    print ( "返回特征名字：\n" , transfer.get_feature_names_out () )
    """
    注：中文(import jieba)：
    data_new = " ".join(list(jieba.cut(data)))
    """

if __name__ == "__main__":
    demo_dict_feature_extraction ()
    demo_text_feature_extraction ()

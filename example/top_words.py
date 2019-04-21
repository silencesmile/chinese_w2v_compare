# -*- coding: utf-8 -*-
# @Time    : 2019/4/20 6:55 PM
# @Author  : yangmingxing
# @File    : top_words.py
# @Software: PyCharm
import gensim
import logging
from gensim import models

def test():
    logging.basicConfig(format="%(asctime)s:%(levelname)s:%(message)s",level=logging.INFO)
    # 250维 加载3个一组的模型文件
    # model = models.Word2Vec.load("../wiki_chinese/model/wiki_corpus.model")

    # 250维  加载独立模型 模型为二进制文件  使用该模型时修改avg_w2v.py 文件中的 51行的模型维度
    model = gensim.models.KeyedVectors.load_word2vec_format("../wiki_chinese/model/wiki_corpus_binary.bin", binary=True)

    #输入一个词找出相似的前10个词
    one_corpus = ["人工智能"]

    # ''' 词库中与one_corpus最相似的10个词'''
    result = model.most_similar(one_corpus[0],topn=10)
    print(result)

    # 两个词的相似度
    # #输入两个词计算相似度
    two_corpus = ["腾讯","阿里巴巴"]
    res = model.similarity(two_corpus[0],two_corpus[1])
    print("similarity:%.4f"%res)

    # 输入三个词类比
    three_corpus = ["北京", "上海", "广州"]
    res = model.most_similar([three_corpus[0], three_corpus[1], three_corpus[2]], topn=100)  # 将返回的结果转换为字典,便于绘制词云
    print(res)



if __name__ == '__main__':
    test()

    # KeyError: "word '报了' not in vocabulary"
    # 错误：“单词'报了'不在词汇中”
# -*- coding: utf-8 -*-
# @Time    : 2019/4/20 5:14 PM
# @Author  : yangmingxing
# @File    : train.py
# @Software: PyCharm

import logging
from gensim.models import word2vec

def main():
    logging.basicConfig(format="%(asctime)s:%(levelname)s:%(message)s",level=logging.INFO)
    sentences = word2vec.LineSentence("../dataset/zhwiki/BB/wiki_corpus")
    # size：单词向量的维度。
    model = word2vec.Word2Vec(sentences,size=250)
    #保存模型  必须3个一起用
    # model.save("../model/wiki_corpus.bin")
    # model.save("../model/wiki_corpus.model")

    # 训练为一个单独二进制压缩文件  可独立使用
    model.save_word2vec_format("../model/wiki_corpus_binary.bin", binary=True)
    # model.wv.save_word2vec_format("./sogou_word2vec/min_count-1/sogou.wor2vec.txt")


if __name__ == "__main__":
    main()

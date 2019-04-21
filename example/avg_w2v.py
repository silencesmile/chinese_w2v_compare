import glob

import sys
import jieba
import numpy as np
import jieba.analyse
import logging

from jieba_tokenizer.tokenizer import JiebaTokenizer
from gensim import models

logger = logging.getLogger(__name__)

sys.path.append("./")

class AvgWord2vec(object):
    """用word2vec+平均词向量的方式生成句子向量"""

    def __init__(self,
                 w2v_model_path="../wiki_chinese/model/wiki_corpus.bin"):

        # self._load_w2v_model(w2v_model_path)
        self.w2v_model = models.Word2Vec.load(w2v_model_path)

        self.tokenizer = JiebaTokenizer()

    def _load_w2v_model(self, path):
        """加载w2v模型"""

        w2v_path = glob.glob("**{}".format(path), recursive=True)
        if len(w2v_path) == 0:
            logger.error("can not find w2v model")
        else:
            logger.info("Loading word2vec model file at {}".format(w2v_path[0]))
            self.w2v_model = models.Word2Vec.load(w2v_path[0], binary=True)
        return

    def transfrom_sentence_to_vec(self, sentence):
        """把句子转换成句子向量"""

        cutted_sentence = self.tokenizer.cut_sentence(sentence)
        # words_list = cutted_sentence.split(" ")
        # print(cutted_sentence)
        vec = self.seg_text_to_vector(cutted_sentence)
        # sentence
        return vec

    def seg_text_to_vector(self, sentence):
        splited_text = sentence.split(" ")
        # size：单词向量的维度。 与训练时保持一致
        vector = np.zeros(250)
        num_of_word = 0
        for word in splited_text:
            try:
                a = self.w2v_model[word]
                for q in a:
                    if np.isnan(q):
                        continue
            except:
                # print(j+"is not in vocabulary")
                continue

            # 不分词语的权重
            # vector += a

            # 词语带权重
            words_weight_dict = self.word_weight_dict(sentence)
            weight = words_weight_dict.get(word)
            if weight is None:
                weight = 1.0
            vector += (a * weight)

            num_of_word += 1

        if (num_of_word == 0) is True:
            return np.zeros(250)
        else:
            vector = vector / num_of_word
            return vector

    def word_weight_dict(self, sentence):
        """根据输入sentence返回一个词权重查询词典"""

        tf_idf_list = jieba.analyse.extract_tags(sentence, topK=None, withWeight=True)
        weight_dict = {}
        for word_weight in tf_idf_list:
            weight_dict[word_weight[0]] = word_weight[1]

        return weight_dict


if __name__ == '__main__':
    sentence = "这是一句测试用的句子，通过这个文件，可以把一句话转换成一个句子向量"
    s2v = AvgWord2vec()
    vec = s2v.transfrom_sentence_to_vec(sentence)
    print(vec)
    exit(0)

# -*- coding: utf-8 -*-
# @Time    : 2019/4/20 8:14 PM
# @Author  : yangmingxing
# @File    : words_com.py
# @Software: PyCharm

from example.avg_w2v import AvgWord2vec
from scipy.spatial.distance import cosine as cos

# 夹角越小越好
best_score = 0.3


def words_base(words_list):
    '''原数据的训练保存'''
    words_infos = []

    for words_info in words_list:
        sentence = words_info.get("words")
        s2v = AvgWord2vec()
        words_vec = s2v.transfrom_sentence_to_vec(sentence)

        words_dict = {"intent": words_info.get("intent"), "words_vec": words_vec}
        words_infos.append(words_dict)

    return words_infos

def words_score(sentence, words_infos):
    '''新数据与老数据对比，分类'''
    s2v = AvgWord2vec()
    words_vec = s2v.transfrom_sentence_to_vec(sentence)

    for words_info in words_infos:
        score = cos(words_vec, words_info.get("words_vec"))
        print(score)

        # 夹角越小越相似
        if score < best_score:
            return words_info.get("intent")

    else:
        return "匹配失败"

if __name__ == '__main__':
    words_list = [
                    {"intent":"天气", "words":"杭州天气怎么样"},
                    {"intent": "年龄", "words": "你今年几岁了"}
                    ]

    words_infos = words_base(words_list)
    # print(words_infos)

    sentence = "北京天气怎么样"
    result = words_score(sentence, words_infos)
    print(result)


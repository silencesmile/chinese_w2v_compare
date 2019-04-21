import glob
import jieba
import logging

logger = logging.getLogger(__name__)

class JiebaTokenizer(object):
    """采用jieba作为分词器"""

    def __init__(self,
                 stop_words_path="/jieba_files/stopwords.txt",
                 costom_dictionary_path="/jieba_files/custom_dictionary.txt"):

        self._load_custom_dictionary(costom_dictionary_path)
        self.stop_words_list = self._load_stop_words(stop_words_path)

    def _load_stop_words(self, filepath):

        jieba_stopwords = glob.glob("**{}".format(filepath), recursive=True)
        if jieba_stopwords is None:
            return []
        else:
            for jieba_stopword in jieba_stopwords:
                logging.info("Loading Jieba stopwords at {}".format(jieba_stopword))
                stopwords = [line.strip() for line in open(jieba_stopword, 'r', encoding='utf-8').readlines()]
                return stopwords

    def _load_custom_dictionary(self, path):
        if path is None:
            return
        else:
            jieba_userdicts = glob.glob("**{}".format(path), recursive=True)
            for jieba_userdict in jieba_userdicts:
                logging.info("Loading Jieba User Dictionary at {}".format(jieba_userdict))
                jieba.load_userdict(jieba_userdict)
            return

    def cut_sentence(self, sentence):
        sentence_seged = jieba.cut(sentence.strip())
        outstr = ''
        for word in sentence_seged:
            if self.stop_words_list is not None and len(sentence) > 2:
                if word not in self.stop_words_list:
                    if word != '\t':
                        outstr += word
                        outstr += " "
            else:
                if word != '\t':
                    outstr += word
                    outstr += " "
        return outstr

if __name__ == '__main__':
    tokenizer = JiebaTokenizer()
    ret = tokenizer.cut_sentence("这是一句话")
    print(ret)

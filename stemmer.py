from tokenizer import SentenceTokenizer
from utils.stopwords import stopwords

import re

class Stemmer():

    def stem_word(self, sentence):
        word_list = sentence.split(' ')
        if not word_list[-1]:
            del(word_list[-1])
        return_list = []
        puctuations = ('.',',','!','?','"',"'",'%','#','@','&')
        suffixes = ['નાં','ના','ની','નો','નું','ને','થી','માં','એ','ઓ','ે','તા','તી','વા','મા','વું','વુ','ો','માંથી','શો','ીશ','ીશું','શે','તો','તું','તાં','્યો','યો','યાં','્યું','યું','્યા','યા','્યાં','સ્વી']
        for word in word_list:
            a = word
            if word.endswith(puctuations):
                a = word[:-1]
            if a in stopwords:
                return_list.append(a)
                continue
            for suffix in suffixes:
                if a.endswith(suffix):
                    a = a.rstrip(suffix)
                    break
            if word.endswith(puctuations):
                a+=str(word[-1])
            return_list.append(a)
        return_sentence = " ".join(return_list)
        return return_sentence


    def stem(self, text):
        l = SentenceTokenizer(text)
        if len(l)==1:
            sentence = l[0]
            return self.stem_word(sentence)
        else:
            a = []
            for sentence in l:
                a.append(self.stem(sentence))
            return a

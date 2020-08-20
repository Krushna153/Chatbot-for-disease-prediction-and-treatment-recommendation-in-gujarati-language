from utils.stopwords import stopwords
import re

def SentenceTokenizer(data):
    data = data.strip()
    data = re.sub(r'([.!?])', r'\1 ', data)
    data = re.split(r'  ',data)
    if not data[-1]:
        del(data[-1])
    return data
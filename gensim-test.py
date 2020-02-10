from gensim.test.utils import datapath
from gensim.models.word2vec import LineSentence,Text8Corpus
from gensim import utils
import gensim.models
import logging
import random
import string
from tqdm import tqdm
from nltk.corpus import brown
def randomString(stringLength=10):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))

# class MyCorpus(object):
#     """An interator that yields sentences (lists of str)."""

#     def __iter__(self):
#         corpus_path = datapath('head500.noblanks.cor')
#         for line in open(corpus_path):
#             # assume there's one document per line, tokens separated by whitespace
#             yield utils.simple_preprocess(line)

def randomCorpus(n_sentences=100000):
    sentences = []
    for i in tqdm(range(n_sentences)):
        sentences.append([randomString(stringLength=4) for i in range(10)])
    return sentences

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
   
    sentences = randomCorpus(n_sentences=1000000)
    model = gensim.models.Word2Vec(workers=8, min_count=1)
    model.build_vocab(sentences)
    model.train(sentences,total_examples=model.corpus_count, epochs=5)
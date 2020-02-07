import re
import unidecode
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import gensim
import nltk
import pandas as pd
from functools import wraps
from time import time


def timeit(method):
    def timed(*args, **kw):
        ts = time()
        result = method(*args, **kw)
        te = time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print ('%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result
    return timed


class Embeddings(object):
    def __init__(self):
        pass

    def word2vec(self, sentences):
        # Create CBOW model 
        cbow = gensim.models.Word2Vec(sentences, min_count = 1,  size = 100, window = 5) 
        # Create SkipGram model 
        skip = gensim.models.Word2Vec(sentences, min_count = 1, size = 100, window = 5, sg = 1) 

        return cbow, skip


class FooModel(object):
    def __init__(self, embedding=CountVectorizer, mod=MultinomialNB):
        self.embedding = embedding()
        self.mod = mod()

    def __repr__(self):
        return 'FooModel: '+str((self.mod.__class__.__name__))+', '+str((self.embedding.__class__.__name__))

    def embed(self, texts) -> ([],[]):
        self.matrix = self.embedding.fit_transform(texts)
        self.headers = self.embedding.get_feature_names()
        return self.matrix, self.headers

    def train(self, X, y) -> None:
        self.mod.fit(X, y)

    def score(self, X, y) -> float:
        return self.mod.score(X, y)

    @timeit
    def predict(self, X) -> ([str],[float]):
        return self.mod.predict(X), self.mod.predict_proba(X)


class FooNLP(object):
    STOPLIST = ['i', 'you', 'am', 'r', 'a', 'an', 'and']

    def __init__(self, model=FooModel(), stoplist=STOPLIST):
        self.model = model
        self.stoplist = stoplist

    def __repr__(self):
        return 'FooNLP: '+self.corpus+', '+str(self.model)

    def full_proc(self, text) -> str:
        text = self.expand(text)
        text = self.clean(text)
        text = self.lemmitize(text)
        text = self.destop(text)
        return text

    def clean(self, text) -> str:
        text = unidecode.unidecode(text)  # clean accents
        text = re.sub(r'[^a-zA-Z\s]', '', text, re.I | re.A)
        text = text.lower()
        text = text.strip()
        return text

    def expand(self, text) -> str:
        text = re.sub(r"i'd", 'i would', text, re.I | re.A)
        text = re.sub(r"i've", 'i have', text, re.I | re.A)
        text = re.sub(r"you've", 'you would', text, re.I | re.A)
        text = re.sub(r"don't", 'do not', text, re.I | re.A)
        text = re.sub(r"doesn't", 'does not', text, re.I | re.A)
        return text

    def lemmitize(self, text) -> str:
        toks = self.tokenize(text)        
        for i, word in enumerate(toks):
            if (len(word) > 4):
                word = re.sub(r"ing\b", '', word, re.I | re.A)
                word = re.sub(r"ed\b", '', word, re.I | re.A)
                word = re.sub(r"s\b", '', word, re.I | re.A)
                toks[i] = word

        return " ".join(toks)

    def lemmitize_word(self, text) -> str:   # not such a good way, complexity merits using nltk lib
        return text

    def destop(self, text,) -> str:
        words = self.tokenize(text)
        return " ".join([x for x in words if x not in self.stoplist])

    def tokenize(self, text) -> [str]:
        toks = text.split(' ')
        return [t.strip() for t in toks if t != '']

    def encode(self, texts) -> []:
        return self.model.embedding.transform(texts)

    def make_embeddings(self, text) -> ([],[]):
        return self.model.embed(text)

    @timeit
    def load_train_stanford(self, samplesize=239232) -> object:
        self.corpus = 'stanford'
        dictfile = 'stanfordSentimentTreebank/dictionary.txt'
        labelfile = 'stanfordSentimentTreebank/sentiment_labels.txt'

        df_dictionary = pd.read_table(dictfile, delimiter='|').sample(samplesize, random_state=5)
        df_labels = pd.read_table(labelfile, delimiter='|').sample(samplesize, random_state=5)
        df_merged = pd.merge(left=df_dictionary, right=df_labels, left_on='id', right_on='id')
        print('merged to corpus size: %d'%len(df_merged))

        # labels need to be changed from float 0.0->1.0 to 5 classes labelea -2,-1,0,1,2 or some strings
        df_merged['labels'] = pd.cut(df_merged['sentiment'], [0.0,0.225,0.45,0.55,0.725,1.1], labels=["real bad", "bad", "medium", "good","real good"])

        # clean and tokenize
        df_merged['text'] = df_merged['text'].apply(lambda row: self.full_proc(row))

        # turn into embeddings
        onehot_dictionary, headers = self.make_embeddings(df_merged['text'].tolist())

        # split sets
        X_train, X_test, y_train, y_test = train_test_split(onehot_dictionary, df_merged['labels'].astype(str), test_size=0.30, random_state=1)

        self.model.train(X_train, y_train)
        print('trained test score: ', self.model, self.model.score(X_test, y_test))
        return self.model
    
    # https://www.kaggle.com/kazanova/sentiment140 - 1.6m tweets
    @timeit
    def load_train_twitter(self, samplesize=1500000) -> object:
        self.corpus = 'twitter'
        dictfile = 'twitter/SentimentAnalysisDataset.csv'

        df_merged = pd.read_table(dictfile, delimiter=',', quotechar='"', error_bad_lines=False).sample(samplesize, random_state=5)
        print('merged to corpus size: %d'%len(df_merged))

        # clean and tokenize
        df_merged['clean_text'] = df_merged['SentimentText'].apply(lambda row: self.full_proc(row))

        # turn into embeddings
        onehot_dictionary, headers = self.make_embeddings(df_merged['clean_text'].tolist())

        # split sets
        X_train, X_test, y_train, y_test = train_test_split(onehot_dictionary, df_merged['Sentiment'].astype(str), test_size=0.30, random_state=1)

        self.model.train(X_train, y_train)
        print('trained test score: ', self.model, self.model.score(X_test, y_test))
        return self.model
    
    def predict(self, X) -> ([str],[float]):
        return self.model.predict(X)

    def score(self, X, y) -> float:
        return self.model.score(X,y)


if __name__ == "__main__":
    nlp = FooNLP()
    nlp2 = FooNLP(model=FooModel(TfidfVectorizer))
    nlp3 = FooNLP(model=FooModel(mod=LogisticRegression))

    nlp.load_train_stanford(5000)
    nlp2.load_train_stanford(5000)
    nlp3.load_train_stanford(5000)
    sents = ['I enjoy happy i love it superstar sunshine','I hate kill die horrible','Do you love or hate me?']
    encoded_vect = nlp.encode(sents)
    encoded_tfid = nlp2.encode(sents)

    print(sents)
    print(encoded_vect)
    print(encoded_tfid)

    # print(smodel.headers)

    print(nlp, nlp.predict(encoded_vect))
    print(nlp2, nlp2.predict(encoded_tfid))
    print(nlp3, nlp3.predict(encoded_vect))

    print('ready for inputs, type ^C or empty line to break out')

    while True:
        txt = input('Enter Text> ')
        if (txt == ''):
            print('quitting see ya')
            break
        encoded_vect = nlp.encode([txt])
        encoded_tfid = nlp2.encode([txt])

        print(nlp.predict(encoded_vect), nlp)
        print(nlp2.predict(encoded_tfid), nlp2)
        print(nlp3.predict(encoded_vect), nlp3)




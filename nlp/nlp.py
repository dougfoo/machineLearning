import re
import os
#import unidecode
import unicodedata
import statistics
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import BernoulliNB, ComplementNB
from sklearn.linear_model import LogisticRegression
import gensim
import pandas as pd
import numpy as np
from functools import wraps
from time import time

#
# publishing as 1.0.0 - 2/18/2020
#


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

class W2VModel(object):
    """
    Wrapper for Word2Vec that creates an internal 2d embedding
    flattened to averages for training and inference
    """
    def __init__(self, mod=LogisticRegression, sg=0, dims=100):
        self.mod = mod()
        self.dims = dims
        self.sg = sg

    def __repr__(self):
        return 'FooModel: '+str((self.mod.__class__.__name__))+', '+str((self.embedding.__class__.__name__))

    def embed(self, texts) -> ([],[]):
        toks = [t.split(" ") for t in texts]    # embed per phrase
        self.embedding = gensim.models.Word2Vec(toks, min_count=1,  size=self.dims, window=5, sg=self.sg)
        self.matrix,self.headers = self.doc_vector(toks)
        return self.matrix, self.headers

    def word_vector(self) -> ([],[]):
        return self.embedding.wv[self.embedding.wv.vocab], self.embedding.wv.vocab

    def doc_vector(self, texts) -> ([],[]):
        v = []
        h = []
        for t in texts:
            doc = [word for word in t if word in self.embedding.wv.vocab]
            if (len(doc)) == 0:
                v.append(np.zeros(self.dims))
            else:
                v.append(np.mean(self.embedding.wv[doc], axis=0))
            h.append(" ".join(doc))
        return v,h

    def train(self, X, y) -> None:
        self.mod.fit(X, y)

    def transform(self, texts) -> []:
        toks = [t.split(" ") for t in texts]
        v, h = self.doc_vector(toks)
        return v

    def score(self, X, y) -> float:
        return self.mod.score(X,y)

    def predict(self, X) -> ([str],[float]):
        return self.mod.predict(X), self.mod.predict_proba(X)


class FooModel(object):
    """
    Basic container for standard 1d embeddings - CountVectorizer, TfidfVectorizer embeddings
    Models MultinomialNB, LogisticRegression that have similar input types
    """
    def __init__(self, mod=BernoulliNB, embedding=CountVectorizer):
        self.embedding = embedding()
        self.mod = mod()

    def __repr__(self):
        return 'FooModel: '+str((self.mod.__class__.__name__))+', '+str((self.embedding.__class__.__name__))

    def embed(self, texts) -> ([],[]):
        self.matrix = self.embedding.fit_transform(texts)
        self.headers = self.embedding.get_feature_names()
        return self.matrix, self.headers

    def word_vector(self) -> ([],[]):
        return self.matrix, self.headers

    def train(self, X, y) -> None:
        self.mod.fit(X, y)

    def score(self, X, y) -> float:
        return self.mod.score(X, y)

    def transform(self, texts) -> []:
        return self.embedding.transform(texts)

    def predict(self, X) -> ([str],[float]):
        # if (isinstance(self.embedding, TfidfVectorizer)):  # or self.mod == GaussianNB
        #     X = X.toarray()
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
        text = self.stem(text)
        text = self.destop(text)
        return text

    def expand(self, text) -> str:
        text = re.sub(r"i'd", 'i would', text, re.I | re.A)
        text = re.sub(r"i've", 'i have', text, re.I | re.A)
        text = re.sub(r"you've", 'you would', text, re.I | re.A)
        text = re.sub(r"don't", 'do not', text, re.I | re.A)
        text = re.sub(r"doesn't", 'does not', text, re.I | re.A)
        return text

    def clean(self, text) -> str:
        def remove_accents(text):
            try:
                text = unicode(text, 'utf-8')
            except (TypeError, NameError): # unicode is a default on python 3 
                pass
            text = unicodedata.normalize('NFD', text)
            text = text.encode('ascii', 'ignore')
            text = text.decode("utf-8")
            return str(text)

        text = remove_accents(text)  # clean accents
        # text = unidecode.unidecode(text)  # clean accents
        text = re.sub(r'[^a-zA-Z\s]', '', text, re.I | re.A)
        text = text.lower()
        text = text.strip()
        return text

    def stem(self, text) -> str:
        toks = self.tokenize(text)        
        for i, word in enumerate(toks):
            if (len(word) > 4):
                word = re.sub(r"ing\b", '', word, re.I | re.A)
                word = re.sub(r"ed\b", '', word, re.I | re.A)
                word = re.sub(r"s\b", '', word, re.I | re.A)
                toks[i] = word
        return " ".join(toks)

    def tokenize(self, text) -> [str]:
        toks = text.split(' ')
        return [t.strip() for t in toks if t != '']

    def destop(self, text,) -> str:
        words = self.tokenize(text)
        return " ".join([x for x in words if x not in self.stoplist])

    def stem_word(self, text) -> str:   # not such a good way, complexity merits using nltk lib
        return text

    def encode(self, texts) -> []:
        return self.model.transform(texts)

    @timeit
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
    def load_train_twitter(self, samplesize=1500000, dictfile='twitter/SentimentAnalysisDataset.csv') -> object:
        self.corpus = 'twitter'

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
    
    @timeit
    def predict(self, X) -> ([str],[float]):
        encoded_X = self.encode(X)
        return self.model.predict(encoded_X)

    @timeit
    def score(self, X, y) -> float:
        encoded_X = self.encode(X)
        return self.model.score(encoded_X,y)

    @timeit
    def save(self, path, obj):
        pickle.dump(obj, open( path, "wb" ) ) 
        print(f'saving... {obj} to {path}')
        return obj

    @timeit
    def load(self, path):
        obj = pickle.load(open(path, "rb")) 
        print(f'loaded... {obj} from {path}')
        return obj

def make_test_model(nlp, sents, label):
    path = f'{label}.ser'
    if (os.path.exists(path)):
        print(f'----- loading model {path}')
        nlp = nlp.load(path)
    else:
        print(f'----- saving model {path}')
        nlp.load_train_twitter(500000)
        nlp.save(path, nlp)   # takes 1min to load, 1.4gb file

    encoded_w2v = nlp.encode(sents)

    pp.pprint(sents)
    print('**--word vectors:')
    wv,wh = nlp.model.word_vector()
    if (isinstance(nlp.model, FooModel)):   # too many samples (1.5m to show), already in row format
        df = pd.DataFrame(wv[0:10].toarray(), columns=wh)
    else:                         # 100 vectors ok, but need Transpose
        df = pd.DataFrame(wv.T, columns=wh)
    pp.pprint(df.head())
    # print('**--sentence vectors:')
    # pp.pprint(nlp.encode(sents))
    print('-----predicts-----')
    # pp.pprint(list(zip(list(zip(*nlp.predict(sents))), sents)))
    p = nlp.predict(sents)
    pp.pprint(list(zip(p[0], list(list(zip(*p[1])))[1])))
    print('\n')
    return nlp

def print_demo():
    nlp = FooNLP(model=W2VModel(mod=LogisticRegression, sg=0, dims=100))   
    m2 = nlp.make_embeddings(sents)
    wv,wh = nlp.model.word_vector()
    df2 = pd.DataFrame(wv.toarray(), columns=wh)
    pp.pprint(sents[1:6])
    pp.pprint(df2.iloc[1:6,52:70])
    pp.pprint(df2.iloc[1:6,52:70])
    pp.pprint(df2.T[['logical','illogical','long','low','luck','made','make','many','may','me','medical','mind','needs']])

if __name__ == "__main__":
    import pprint
    import pandas as pd
    import numpy as np
    from nlp import FooNLP, FooModel, W2VModel
    pp = pprint.PrettyPrinter(width=140)
    pd.set_option('precision', 2)
    np.set_printoptions(precision=2)

    sents = [
        'I fail to comprehend your indignation, sir. I have simply made the logical deduction that you are a liar',
        'The needs of the many outweigh the needs of the few, or the one',
        'Live long and prosper',
        'It would be illogical to kill without reason',
        'I''ll never understand the medical mind',
        'It is curious how often you humans manage to obtain that which you do not want',
        'Computers make excellent and efficient servants, but I have no wish to serve under them',
        'Captain, you almost make me believe in luck',
        'My congratulations, Captain—a dazzling display of logic',
        'I have never understood the female capacity to avoid a direct answer to any question',
        'After a time, you may find that having is not so pleasing a thing after all as wanting. It is not logical, but is often true',
        'I''m frequently appalled by the low regard you Earthmen have for life',
        'Has it occurred to you that there is a certain...inefficiency in constantly questioning me on things you''ve already made up your mind about?'
        ]

    nlp1 = FooNLP(model=W2VModel(mod=LogisticRegression, sg=0, dims=100))   
    nlp1 = make_test_model(nlp1, sents, 'w2vcbow.lr.fulltwitter.foonlp') 

    # nlp2 = FooNLP(model=W2VModel(mod=BernoulliNB, sg=0, dims=100))   
    # nlp2 = make_test_model(nlp2, sents, 'w2vcbow.nb.fulltwitter.foonlp') 

    # nlp3 = FooNLP(model=W2VModel(mod=LogisticRegression, sg=1, dims=100))   
    # nlp3 = make_test_model(nlp3, sents, 'w2vsg.lr.fulltwitter.foonlp') 

    # nlp4 = FooNLP(model=W2VModel(mod=BernoulliNB, sg=1, dims=100)) 
    # nlp4 = make_test_model(nlp4, sents, 'w2vsg.nb.fulltwitter.foonlp') 

    nlp5 = FooNLP(model=FooModel(mod=BernoulliNB, embedding=TfidfVectorizer) ) # need NB that works w/ sparse
    nlp5 = make_test_model(nlp5, sents, 'tfidf.nb.fulltwitter.foonlp')        

    # nlp6 = FooNLP(model=FooModel(mod=BernoulliNB, embedding=CountVectorizer))  # need NB that works w/ sparse 
    # nlp6 = make_test_model(nlp6, sents, 'cvec.nb.fulltwitter.foonlp')

    # nlp7 = FooNLP(model=FooModel(mod=LogisticRegression, embedding=TfidfVectorizer) ) # works w/o transform
    # nlp7 = make_test_model(nlp7, sents, 'tfidf.lr.fulltwitter.foonlp')

    # nlp8 = FooNLP(model=FooModel(mod=LogisticRegression, embedding=CountVectorizer))  # works w/o transform
    # nlp8 = make_test_model(nlp8, sents, 'cvec.lr.fulltwitter.foonlp')


    pp.pprint('ready for inputs, type ^C or empty line to break out')

    while True:
        txt = input('Enter Text> ')
        if (txt == ''):
            pp.pprint('quitting see ya')
            break
        print(nlp1.predict([txt]), txt, nlp1.model)
        print(nlp2.predict([txt]), txt, nlp2.model)
        print(nlp3.predict([txt]), txt, nlp3.model)
        print(nlp4.predict([txt]), txt, nlp4.model)
        print(nlp5.predict([txt]), txt, nlp5.model)
        print(nlp6.predict([txt]), txt, nlp6.model)
        print(nlp7.predict([txt]), txt, nlp7.model)
        print(nlp8.predict([txt]), txt, nlp8.model) 
        print('\n')




import re
import unidecode
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
import pandas as pd


class FooNLP(object):
    STOPLIST = ['i', 'you', 'am', 'r', 'a', 'an', 'and']

    def __init__(self, txt='', stoplist=STOPLIST):
        self.stoplist = stoplist
        self.original_txt = txt
        self.cleaned_txt = self.full_proc(txt)
        self.cleaned_tokens = self.tokenize(self.cleaned_txt)

    def full_proc(self, text):
        text = self.expand(text)
        text = self.clean(text)
        text = self.lemmitize(text)
        text = self.destop(text)
        return text

    def clean(self, text):
        text = unidecode.unidecode(text)  # clean accents
        text = re.sub(r'[^a-zA-Z\s]', '', text, re.I | re.A)
        text = text.lower()
        text = text.strip()
        return text

    def expand(self, text):
        text = re.sub(r"i'd", 'i would', text, re.I | re.A)
        text = re.sub(r"i've", 'i have', text, re.I | re.A)
        text = re.sub(r"you've", 'you would', text, re.I | re.A)
        text = re.sub(r"don't", 'do not', text, re.I | re.A)
        text = re.sub(r"doesn't", 'does not', text, re.I | re.A)
        return text

    def lemmitize(self, text):
        toks = self.tokenize(text)
        return " ".join([self.lemmitize_word(x) for x in toks])

    def lemmitize_word(self, text):   # not such a good way
        if (len(text) > 4):
            text = re.sub(r"ing\b", '', text, re.I | re.A)
            text = re.sub(r"ed\b", '', text, re.I | re.A)
            text = re.sub(r"s\b", '', text, re.I | re.A)
        return text

    def destop(self, text, stoplist=STOPLIST):
        words = self.tokenize(text)
        return " ".join([x for x in words if x not in stoplist])

    def tokenize(self, text):
        toks = text.split(' ')
        return [t.strip() for t in toks if t != '']

    def bag_of_words(self, texts, ngram_min=1, ngram_max=1):
        cv = CountVectorizer(min_df=0.0, max_df=1.0, ngram_range=(ngram_min, ngram_max))
        cm = cv.fit_transform(texts.tolist())
        matrix = cm.toarray()
        headers = cv.get_feature_names()
        return headers, matrix

    def tfidf(self, texts):
        tv = TfidfVectorizer(min_df=0.0, max_df=1.0, use_idf=True)
        tm = tv.fit_transform(texts)
        matrix = tm.toarray()

        headers = tv.get_feature_names()
        return headers, matrix

    def cosine_sim(self, texts):
        tv = TfidfVectorizer(min_df=-0.0, max_df=1.0, use_idf=True)
        tm = tv.fit_transform(texts)
        sm = cosine_similarity(tm)
        return sm

    def train_naivebayes(self):
        "'"
        # load
        df_dictionary = pd.read_table('stanfordSentimentTreebank/dictionary_sm.txt', delimiter='|')
        df_labels = pd.read_table('stanfordSentimentTreebank/sentiment_labels_sm.txt', delimiter='|')
        df_merged = pd.merge(left=df_dictionary, right=df_labels, left_on='id', right_on='id')

        # labels need to be changed from float 0.0->1.0 to 5 classes labelea -2,-1,0,1,2 or some strings
        df_merged['labels'] = pd.cut(df_merged['sentiment'], [0.0,0.2,0.4,0.6,0.8,1.1], labels=["real bad", "bad", "medium", "good","real good"])

        # clean and tokenize
        df_merged['text'] = df_merged['text'].apply(lambda row: self.full_proc(row))

        # turn into embeddings
        headers, onehot_dictionary = self.bag_of_words(df_merged['text'])

        # split sets
        X_train, X_test, y_train, y_test = train_test_split(onehot_dictionary, df_merged['labels'], test_size=0.30, random_state=1)

        # train model
        clf = MultinomialNB()
        clf.fit(X_train, y_train)

        # test model
        print(clf.predict(X_test))
        print(y_test)

        return clf, headers    # trained model and header sequence


if __name__ == "__main__":
    nlp = FooNLP()
    sentences = ['The indian life of the indian pi', 'The life and pain of the french fiancée', 'my life my death my pain']
    print(sentences)
    sm = nlp.cosine_sim(sentences)
    print(sm)

    model, headers = nlp.train_naivebayes()
    print(model)

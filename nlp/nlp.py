import re
import unidecode
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


class FooNLP(object):
    STOPLIST = ['i', 'you', 'am', 'r', 'a', 'an', 'and']

    def __init__(self, txt='', stoplist=STOPLIST):
        self.stoplist = stoplist
        self.original_txt = txt
        ctxt = txt
        ctxt = self.expand(ctxt)
        ctxt = self.clean(ctxt)
        ctxt = self.lemmitize(ctxt)
        ctxt = self.destop(ctxt)
        self.cleaned_tokens = self.tokenize(ctxt)
        self.cleaned_txt = " ".join(self.cleaned_tokens)

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
        cm = cv.fit_transform(texts)
        matrix = cm.toarray()
        headers = cv.get_feature_names()
        return headers, matrix

    def tfidf(self, texts):
        tv = TfidfVectorizer(min_df=0.0, max_df=1.0, use_idf=True)
        tm = tv.fit_transform(texts)
        matrix = tm.toarray()

        headers = tv.get_feature_names()
        return headers, matrix


if __name__ == "__main__":
    nlp = FooNLP()
    sentences = ['The indian life of the indian pi', 'The life and pain of the french fiancée', 'my life my death my pain']
    (h, m) = nlp.tfidf(sentences)
    df = pd.DataFrame(m, columns=h)
    print(sentences)
    print(df)
    print(df['indian'].sum() )
    print(df['death'].sum() )
    print(df['the'].sum() )
    print(df['life'].sum() )
    # print(df['the indian'].sum() )
    # print(df['the indian pi'].sum() )
    
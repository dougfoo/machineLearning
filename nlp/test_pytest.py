from nlp import FooNLP, FooModel, W2VModel
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd


test_inputs = [
    "The life  of π",
    "My fiancée is  > 30",
    "(I) love being happy",
    "5 + 8 = 12 does that mean anything",
    "I don't wanna go on with u like that",
    "Fractions 3/4 75% is meaningful",
]

clean_outputs = [
    "the life of p",
    "my fiancee is",
    "love be happy",
    "does that mean anyth",  # note the tricky part of stemming
    "do not wanna go on with u like that",
    "fraction is meaningful",
]

def test_embeddings():
    emb = W2VModel()
    sentences = [['the','sentence','is','red'],['my','sentence','is','blue','zeta'], ['my','sentence','is','not','gamma']]
    v, h = emb.embed(sentences)
    print(v)
    print(h)


def test_clean():
    nlp = FooNLP()
    assert nlp.clean("The life of π") == "the life of p"
    assert nlp.clean("my fiancée is >  30") == "my fiancee is"


def test_destop():
    nlp = FooNLP()
    assert nlp.destop('what a world') == 'what world'


def test_tokenize():
    nlp = FooNLP()
    assert " ".join(nlp.tokenize('what a   world')) == 'what a world'


def test_lemmitize():
    nlp = FooNLP()
    assert nlp.lemmitize('worked at happening places') == 'work at happen place'


def test_bag_of_words():
    nlp = FooNLP()
    sentences = ['The indian life of the indian pi', 'The life and pain of the french fiancée', 'my life my death my pain']
    (m, h) = nlp.make_embeddings(sentences)
    df = pd.DataFrame(data=m.toarray(), columns=h)
    print(sentences)
    print(df)
    assert df['indian'].sum() == 2
    assert df['death'].sum() == 1
    assert df['the'].sum() == 4
    assert df['life'].sum() == 3
    # assert df['the indian'].sum() == 2   -- can't pass , ngram_max=2 yet
    assert 'the indian pi' not in df.columns 


def test_tfidf():
    nlp = FooNLP(FooModel(TfidfVectorizer))
    sentences = ['The indian life of the indian pi', 'The life and pain of the french fiancée', 'my life my death my pain']
    (m, h) = nlp.make_embeddings(sentences)
    df = pd.DataFrame(m.toarray(), columns=h)
    print(sentences)
    print(h)
    print(df)
    assert round(df['indian'].sum(),2) == 0.70
    assert round(df['death'].sum(),2) == 0.30
    assert round(df['the'].sum(),2) == 1.11
    assert round(df['life'].sum(),2) == 0.61


def test_all():
    for input, output in zip(test_inputs, clean_outputs):
        nlp = FooNLP()
        assert nlp.full_proc(input) == output

def test_stanford_countv():
    nlp = FooNLP()
    smodel = nlp.load_train_stanford(5000)
    sents = ['I am so happy i love it super','I hate kill die horrible','Do you love or hate me?']
    encoded_sents = nlp.encode(sents)
    r,p = smodel.predict(encoded_sents)
    assert (r[0] == 'bad')
    assert (r[1] == 'medium')
    assert (r[2] == 'medium')

def test_stanford_tfidf():
    nlp = FooNLP(model=FooModel(TfidfVectorizer))
    smodel = nlp.load_train_stanford(50000)
    sents = ['I am so happy i love it super','I hate kill die horrible','Do you love or hate me?']
    encoded_sents = nlp.encode(sents)
    r,p = smodel.predict(encoded_sents)
    assert (r[0] == 'good')
    assert (r[1] == 'good')
    assert (r[2] == 'good')

def test_word2vec():
    w2v = W2VModel()
    sents = ['I am so happy i love it super','I hate kill die horrible','Do you love or hate me?']
    v,h = w2v.embed(sents)

    print(v)
    print(h)

    assert(1==1)



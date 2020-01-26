import re
import unidecode


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


if __name__ == "__main__":
    nlp = FooNLP()
    assert nlp.clean("The life of π") == "the life of"
    assert nlp.clean("my fiancée") == "my fiancee"

from nlp import FooNLP

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


def test_all():
    for input,output in zip(test_inputs, clean_outputs):
        nlp = FooNLP(input)
        assert nlp.cleaned_txt == output
from myutils import *
from gdsolvers import *
import inspect

def test_dummy():
    print (inspect.currentframe().f_code.co_name)
    assert (1+3) == 4

def test_get_gaga_data():
    print (inspect.currentframe().f_code.co_name)   
    d = getGagaData(maxrows=500,maxfeatures=8000,gtype=1,stopwords=None)
    assert(d[0].shape == (176, 3652))
    d = getGagaData(maxrows=500,maxfeatures=8000,gtype=0,stopwords=None)
    assert(d[0].shape == (129, 3855))
    d = getGagaData(maxrows=500,maxfeatures=8000,gtype=None,stopwords=None)
    assert(d[0].shape == (305, 5938))
    d = getGagaData(maxrows=500,maxfeatures=8000,gtype=None,stopwords='english')
    assert(d[0].shape == (305, 5690))

if __name__ == "__main__":
    log.getLogger().setLevel(log.WARN)
    test_get_gaga_data()


import requests, pandas, io, numpy, argparse, math
import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
import featureEngineering as fe
from sympy.core.compatibility import as_int
import sympy.concrete.summations as sum
from myutils import *
from sklearn.utils import shuffle
from mpmath import *
import logging as log

def getTestData():
    df = pandas.read_csv('fakeGagaData.dat')
    return df


if __name__ == "__main__":
    log.basicConfig(level=log.WARN)
    log.info('start %s'%(log.getLogger().level))


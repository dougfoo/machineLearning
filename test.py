from myutils import *

# fake commit 2

print 'x'
print('c')


def toMatrixTest(df):
    x,y = sp.symbols('x y')
    X = df.iloc[:,1:6]
    Y = df['gaga']
    print('pre matrix X,Y',X,Y)
    X = numpy.asmatrix(X.as_matrix())
    Y = Y.as_matrix()
    print('post as matrix X,Y',X,Y)
    return X,Y

def toSymbolsTest(df):
    g,h,y = sp.symbols('g h y')
    t = sp.symbols('t0 t1 t2 t3 t4 t5')
    x = sp.symbols('x0 x1 x2 x3 x4 x5')
#    h = t[0]*x[0] + t[1]*x[1] + t[2]*x[2] + t[3]*x[3] + t[4]*x[4] + t[5]*x[5] 
    h = t[0]*x[0] + t[1]*x[1]
    g = 1 / (1+mp.e**-h)   # wrap h in sigmoid

    print('x:',x)
    print('t:',t)
    print('h:',h)
    print('g:',g)

    guesses = [1,1]
    evalPartialDeriv(g, t[0], t[0:3], x[0:3], df, guesses)

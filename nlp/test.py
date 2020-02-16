import pickle

class Bar(object):
    def __init__(self, var1=1, var2='x'):
        self.var1 = var1
        self.var2 = var2

    def __repr__(self):
        return f'Bar {self.var1} - {self.var2}'


class Foo(object):
    def __init__(self, var3=3):
        self.var3 = var3
        self.bar = Bar(2,'y')

    def printme(self):
        print(f'var1: {self.bar.var1}, var2: {self.bar.var2}, var3: {self.var3}')

    def save(self, path, obj):
        pickle.dump(obj, open( path, "wb" ) ) 
        print(f'saving... {obj} to {path}')
        return obj

    def load(self, path):
        obj = pickle.load(open(path, "rb")) 
        print(f'loaded... {obj} from {path}')
        return obj

import io
f = Foo(9)
f.printme()
f.save('test.ser', f)
o = f.load('test.ser')
print(type(o))
o.printme()
print(o.bar)

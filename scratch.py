
class Clazz(object):
    def __init__(self):
        print ('__init__',self)

    def foo(self):
        print('Clazz::foo')

    def __call__(self, a):
        print('Clazz:call', a)

print('start')
c = Clazz()
print('class',c)
c.foo()
c('test')

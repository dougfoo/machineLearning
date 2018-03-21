from sympy import *
x, y, z = symbols('x y z')
init_printing(use_unicode=True)

print 'ab'
print 1+2

msg = "Hello World"
print(msg)

A,B = symbols('A B')
print (diff(A*x+B, A))
print (diff(A*x+B, B))



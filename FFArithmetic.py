# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np

class DomainElement(object):
    def __radd__(self, other): return self + other

    def __rsub__(self, other): return -self + other

    def __rmul__(self, other): return self * other


class FieldElement(DomainElement):
    def __truediv__(self, other): return self * other.inverse()

    def __rtruediv__(self, other): return self.inverse() * other

    def __div__(self, other): return self.__truediv__(other)

    def __rdiv__(self, other): return self.__rtruediv__(other)


#Cast "self" to same type as "other".
def typecheck(f):
    def newF(self, other):
        if type(self) is not type(other):
            try:
                other = self.__class__(other)
            except TypeError:
                message = 'Not able to typecast %s of type %s to type %s in function %s'
                raise TypeError(message % (other, type(other).__name__, type(self).__name__, f.__name__))
            except Exception as e:
                message = 'Type error on arguments %r, %r for functon %s. Reason:%s'
                raise TypeError(
                    message % (self, other, f.__name__, type(other).__name__, type(self).__name__, e))

        return f(self, other)
    return newF

#Multiply, while subsequently reducing modulo m. Removes numerical errors when factors are huge.
def multiply(a,b,m):
    res = 0
    while b >0:
        if b % 2:
            res = (res+a) % m
        a = (a*2) % m
        b = int(b/2)
    return res % m

def GF(p):
    class IntegerModP(FieldElement):
        def __init__(self, n):
            self.n =n % p
            self.field = IntegerModP

        @typecheck
        def __add__(self, other): return IntegerModP(self.n + other.n)

        @typecheck
        def __sub__(self, other): return IntegerModP(self.n - other.n)

        @typecheck
        def __mul__(self, other): return IntegerModP(multiply(self.n, other.n, self.p))

        @typecheck
        def power(self, other): return IntegerModP(pow(self.n, other.n,self.p))

        @typecheck
        def __truediv__(self, other): return self.n * other.inverse()

        @typecheck
        def __div__(self, other): return self.n * other.inverse()

        def __neg__(self): return IntegerModP(-self.n)

        @typecheck
        def __eq__(self, other): return isinstance(other, IntegerModP) and self.n == other.n

        def __abs__(self): return abs(self.n)

        def __str__(self): return str(self.n)

        def __repr__(self): return '%d' % (self.n)

        def __divmod__(self, divisor):
            q, r = divmod(self.n, divisor.n)
            return (IntegerModP(q), IntegerModP(r))


        def extendedEuclideanAlgorithm(self, a, b):
            if abs(b) > abs(a):
                (x, y, d) = self.extendedEuclideanAlgorithm(b, a)
                return (y, x, d)

            if abs(b) == 0:
                return (1, 0, a)

            x1, x2, y1, y2 = 0, 1, 1, 0
            while abs(b) > 0:
                q, r = divmod(a, b)
                x = x2 - q * x1
                y = y2 - q * y1
                a, b, x2, x1, y2, y1 = b, r, x1, x, y1, y

            return (x2, y2, a)

        def inverse(self):
            x, y, d = self.extendedEuclideanAlgorithm(self.n, self.p)
            return IntegerModP(x)

        @classmethod
        def random_element(self):
            return IntegerModP(np.random.randint(0, self.p, dtype = np.int64))

    IntegerModP.p = p
    IntegerModP.__name__ = 'Z/%d' % (p)
    return IntegerModP

#F = GF(7)#(7979490791)
#
#a = F(10)
#b = F(10)
##
#print (a+b)
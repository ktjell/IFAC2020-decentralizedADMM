# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 12:56:38 2018

@author: kst
"""

import FFArithmetic as field
import numpy as np
#np.random.seed(1)
# Creates shares of secrets using Shamir's secret sharing scheme.
def share(F, x, t, n):
    shares = []
    c = [F.random_element() for i in range(t)]
    for i in range(1, n + 1):
        s = x
        for j in range(1, t + 1):
            s += c[j - 1] * F.power(F(i),j)
        shares.append(s)
    s = np.array(shares)
    return np.array(shares)


# Creates the "recombination"-vector used to reconstruct a secret from its shares.
def basispoly(F,n):
    r = []
    C = range(1, n + 1)

    for i in range(1, n + 1):
        c = [k for k in C if k != i]
        p = 1
        for j in range(n - 1):
            p *= -F(c[j]) / (F(i) - F(c[j]))
        r.append(p)
    return r

# reconstruct secret.
def rec(F,x):
    res = F(0)
    n = len(x)
    y = basispoly(F,n)
    for i in range(len(x)):
        res += x[i] * y[i]
    return res


#F = field.GF(97)
#t = 3
#n = 4
#xs = share(F, 5, t, n)
#ys = share(F,2, t,n)
#
#ss = xs + ys
#print(rec(F,ss))
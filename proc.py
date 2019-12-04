# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 14:29:02 2018

@author: kst
"""

import numpy as np
import FFArithmetic as field
import shamir_scheme as ss

#Secure mulitplication of two secrets. -Resulting polynomial is still degree t.
def mult(F,a,b,t,n):
    r = ss.basispoly(F,n)
    h = a*b
    hs = [] 
    for i in range(n):
        hs.append(ss.share(F,h[i],t,n))
    s = ss.share(F,0,t,n)
    for i in range(n):
        s+= np.array(r[i]) * np.array(hs[i])
    return np.array(s) 

#Create a Beaver's triplet.
def triplet(F, n, t):
    matA = []
    matB = []
    for i in range(n):
        a = F.random_element()
        b = F.random_element()
        matA.append(ss.share(F, a, t, n))
        matB.append(ss.share(F, b, t, n))
    matrixA = np.array(matA)
    matrixB = np.array(matB)
    
    sharesA = np.sum(matrixA, 0)
    sharesB = np.sum(matrixB, 0)
    
    sharesC = mult(F, sharesA,sharesB, t,n)
    
    return [[int(str(i)),int(str(j)),int(str(k))] for i,j,k in zip(sharesA,sharesB,sharesC)]

def randomBitsDealer(F,n,t,l):
    bits = [np.random.choice([1,0]) for i in range(l)]
    bits.insert(0, 0)
    r = ''
    for i in bits:
        r+= str(i)
    rshares = ss.share(F,int(r,2), t,n)
    bitshares = [ss.share(F,i, t, n) for i in bits]
    res = []
    for i in range(n):
        temp = []
        for j in range(len(bits)):
            temp.append(bitshares[j][i])
        res.append(temp)
    return rshares, res




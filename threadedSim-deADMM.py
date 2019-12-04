# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 14:47:14 2018

@author: kst
"""
import numpy as np
from threading import Thread
import FFArithmetic as field
import shamir_scheme as ss
import proc
import time
import queue as que
from ipcon1 import ipconfigs as ips
from ipcon1 import network as nw
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class server:
    securecom = {}
    broadcasts = {}
    def __init__(self,F, n, t, numTrip, l = 7):
        self.b = ss.share(F,np.random.choice([-1,1]), t, n)
        self.triplets = [proc.triplet(F,n,t) for i in range(numTrip)]
        self.r, self.rb = proc.randomBitsDealer(F,n,t,l)

class communicationSimulation:
    def __init__(self,q):
        self.q = q
        
    def com(self,add, val):
        index = ips.party_addr.index(add)
        if not self.q[index].full():
            self.q[index].put(val)
    
class party(Thread):
    party_addr = ips.party_addr
    def __init__(self, F, x, n, t, i, q, com, E, Q, f, retur):
        Thread.__init__(self)
        self.c = 0
        self.comr = 0
        self.F = F
        self.f = f
        self.E = E
        self.Q = Q
        self.x = x
        self.n = n
        self.t = t
        self.i = i
        self.q = q
        self.com = com
        self.retur = retur
        self.comr = 0        
        self.comtime = 0
        self.recv = {}
        
    def readQueue(self):
        while not self.q.empty():
            b = self.q.get()
            self.recv[b[0]] = b[1]
#            self.q2.put([b[0][-1], b[1]])
            
    def broadcast(self, name, s):
        for i in nw.N[self.i]:
            self.com.com(self.party_addr[i], [name + str(self.i), s])
                    
    
    def get_shares(self, name):
        res = []
        for i in nw.N[self.i]:
            while name + str(i) not in self.recv:
                self.readQueue()    
            res.append(self.recv[name+str(i)])
            del self.recv[name + str(i)]
        return res
    
        
    def run(self):
        ite = 30

#        print('starting party ', self.i)
#        self.get_triplets()
#        self.tt = self.get_share('b')
        c = 1
        x = 0
        v = np.zeros((2,1))
        p = np.zeros((2,1))
        ln = len(nw.N[self.i])
        xx = []

        for j in range(ite):
## DISTRIBUTE INPUT            
            self.broadcast('v'+str(self.comr), v)
            
## GET OTHER AGENTS DATA            
            ss=self.get_shares('v'+str(self.comr))
#            print(self.i, np.array(sum(ss)).flatten())

            sm = ln*v - sum(ss)              #sum([v - i for i in ss]) 
            sp = ln*v + sum(ss)                     #sum([v + i for i in ss]) 
#            if self.i == 1:
#                print(list(sm) , list(),'\n')
#                print(list(sp), list(),'\n')
#            if self.i ==2:
##                print('ss',ss)
#                print('ss',list(sum(ss)), '\n')
                
            self.comr+=1
#            s = np.sum(ss)
#            print('Party {} done with com in ite {}.'.format(self.i, j))

## START MINIMIZATION
            p = p + c*sm#( ln* v - s)
            g = lambda x: self.f(x) + (c/(4*ln)) * np.linalg.norm(  
                (1./c) * (np.dot(self.E, x) - (1./self.n)*self.Q) - (1/c) * p + sp    # ln*v + s
                ,2)**2
            res = minimize(g, x0=0)
            x = res['x'][0]
            xx.append(x)

            v = (1./(2*ln)) * (sp #(ln*v+s)
                     -(1./c)*p 
                    + (1./c)* (np.dot(self.E, x)- (1./self.n)*self.Q)
                    )

#            line1.set_ydata(xx)
#            fig.canvas.draw()
#            fig.canvas.flush_events()
#        print('Party {} is finished!'.format(self.i))     
        self.retur.put(xx)
        print('x:',x)

#  7979490791   
F = field.GF(97)            
n = 6
t = 1
m = 2 #number of constraints
np.random.seed(1)
f = []
for k in range(n):
    f.append(lambda x: ((x-2)**2))

E = []
for k in range(n):
    E.append(np.random.randint(10, size = m).reshape((2,1)))
    
Q = np.random.randint(10,size=m).reshape((2,1))    

q = [que.Queue(),que.Queue(),que.Queue(),que.Queue(),que.Queue(),que.Queue()]

retur = que.Queue()
 
com = communicationSimulation(q)


threads = []
for k in range(n):
    threads.append(party(F,2,n,t,k,q[k],com,E[k], Q, f[k], retur))

#start = time.time()
    
for t in threads:
    t.start()


for t in threads:
    t.join()
plt.figure('2')
#print('HERE')
for i in range(n):
    plt.plot(retur.get())

#while retur.not_empty:
#    plt.plot(retur.get())

#end = time.time()
#ex = end-start
#print('Execution time: ', ex)

#print(p1.comtime)
#print(p2.comtime)
#print(p3.comtime)
#print('\n')
#print(p1.comtime/ex)
#print(p2.comtime/ex)
#print(p3.comtime/ex)

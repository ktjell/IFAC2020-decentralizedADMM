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
from ipcon import ipconfigs as ips
from ipcon import network as nw
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
    
    def calCliqSum(self,val):
        #### YOU NEED TO GENERALIZE THIS TO VAL BEING A VECTOR!! AND ALSO WHAT HAPPENS WHEN VAL NEEDS TO BE IN F????
        for i in nw.C[self.i]:
            r1 = []
            for j in range(len(i)-2):
                r1.append([self.F.random_element() for i in range(len(val))])
            r2=np.array(r1)
            r1.append(list(-sum(r2)))   
            for k,j in enumerate(i[1:]):
                self.com.com(self.party_addr[j], ['r'+str(i[0]) + str(self.i), r1[k]])

        rec = []
        for i in nw.C[self.i]:
            res1 = []
            for j in i[1:]:
                while 'r' + str(i[0]) + str(j) not in self.recv:
                    self.readQueue()    
                res1.append((self.recv['r'+str(i[0])+str(j)]))
                del self.recv['r'+str(i[0]) + str(j)]
            rec.append(sum(np.array(res1)))

        for i,ii in enumerate(nw.C[self.i]):
            for k,j in enumerate(ii[1:]):
                m = np.array(val/nw.W[self.i][i][k], dtype=np.int) + np.array(rec[i])
                self.com.com(self.party_addr[j],
                               ['val' +str(ii[0]) + str(self.i), m ])

        res = []
        for i in nw.C[self.i]:
            res1 = []
            for j in i[1:]:
                while 'val'+str(i[0]) + str(j) not in self.recv:
#                    if self.i == 5:
#                        print(self.recv)
                    self.readQueue()    
                res1.append((self.recv['val'+str(i[0])+str(j)]))
                del self.recv['val'+str(i[0]) + str(j)]
            res.append((np.array(res1)))
   
        return res
    
        
    def run(self):
        ite = 3
        scale = 1000
        print('starting party ', self.i)
        c = 1
        x = 0
        v = np.zeros(2, dtype=np.int)
        v1 = np.zeros(2, dtype=np.int)
        p = np.zeros((2,1))
        ln = len(nw.N[self.i])
        xx = []

        for j in range(ite):

## DISTRIBUTE INPUT
        ## DISTRIBUTE INPUT            
            res = self.calCliqSum(v1)
            s= [sum(i) for i in res]  # sum contributions from each agent in each clique
#            print('calculated sum for agent {}: {}'.format(self.i, sum(s)))
## GET OTHER AGENTS DATA
            
                    
                
            sss = sum(s) # sum contribution from each clique
            print(self.i, sss)
            ss = []
            for i in sss:
                if int(str(i)) > F.p/2:
                    ss.append((int(str(i)) - F.p) / scale)
                else:
                    ss.append(int(str(i))/scale)
            
#            ss = np.array([int(str(i))/scale for i in ss])
#            if self.i == 2:
#                print('ss',ss,'\n')
            sm =(ln*v - ss).reshape((2,1))            #sum([v - i for i in ss]) 
            sp =(ln*v + ss).reshape((2,1))                 #sum([v + i for i in ss]) 

#            
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
            v = v.reshape((2,))
            vv = [int(i*scale) for i in v]
            v1 = np.array(vv)
#            if self.i == 2:
#                print(v,'\n')
#                print(sp,'\n')
#            line1.set_ydata(xx)
#            fig.canvas.draw()
#            fig.canvas.flush_events()
#        print('Party {} is finished!'.format(self.i))     
        self.retur.put(xx)
        
        
        
        print('x:',x)

mm=  7979490791   
F = field.GF(mm)            
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
    threads.append(party(F,np.array([2*k+2,2*k+2, 2*k+2, 2*k+2]),n,t,k,q[k],com,E[k], Q, f[k], retur))

#start = time.time()
    
for t in threads:
    t.start()


for t in threads:
    t.join()

#print('HERE')'
plt.figure('1')
for i in range(n):
#    if retur.not_empty == 1:
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

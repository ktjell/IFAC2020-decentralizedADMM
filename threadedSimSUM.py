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
    def __init__(self, F, x, n, t, i, q, com):
        Thread.__init__(self)
        self.c = 0
        self.comr = 0
        self.F = F
        self.x = x
        self.n = n
        self.t = t
        self.i = i
        self.q = q
        self.com = com
        
        self.comtime = 0
        self.recv = {}
        
    def readQueue(self):
        while not self.q.empty():
            b = self.q.get()
            self.recv[b[0]] = b[1]
#            self.q2.put([b[0][-1], b[1]])
    
    def calCliqSum(self,val):

        for i in nw.C[self.i]:
            r1 = []
            for j in range(len(i)-2):
                r1.append(self.F.random_element())
            r1.append(-sum(r1))    
            for k,j in enumerate(i[1:]):
                self.com.com(self.party_addr[j], ['r'+str(i[0]) + str(self.i), int(str(r1[k]))])
        
        rec = []
        for i in nw.C[self.i]:
            res1 = []
            for j in i[1:]:
                while 'r' + str(i[0]) + str(j) not in self.recv:
                    self.readQueue()    
                res1.append(self.F(self.recv['r'+str(i[0])+str(j)]))
                del self.recv['r'+str(i[0]) + str(j)]
            rec.append(sum(res1))
        
        for i,ii in enumerate(nw.C[self.i]):
            for k,j in enumerate(ii[1:]):
                self.com.com(self.party_addr[j],
                               ['val' +str(ii[0]) + str(self.i), int(str(int(val/nw.W[self.i][i][k]))) + int(str(rec[i]))])
        
        res = []
        for i in nw.C[self.i]:
            res1 = []
            for j in i[1:]:
                while 'val'+str(i[0]) + str(j) not in self.recv:
#                    if self.i == 5:
#                        print(self.recv)
                    self.readQueue()    
                res1.append(self.F(self.recv['val'+str(i[0])+str(j)]))
                del self.recv['val'+str(i[0]) + str(j)]
            res.append(sum(res1))
        
        return res
    
    def run(self):
## DISTRIBUTE INPUT
        ## DISTRIBUTE INPUT            
            res = self.calCliqSum(self.x)
            
## GET OTHER AGENTS DATA            
            s = sum(res) #- self.x
            
            print('calculated sum for agent {}: {}'.format(self.i, s))

        

#  7979490791   
F = field.GF(97)            
n = 3
t = 1

q = [que.Queue(),que.Queue(),que.Queue(),que.Queue(),que.Queue(),que.Queue()]

com = communicationSimulation(q)


p1 = party(F,2, n,t,0, q[0], com)
p2 = party(F,4, n,t,1, q[1], com)
p3 = party(F,6, n,t,2, q[2], com)
p4 = party(F,8, n,t,3, q[3], com)
p5 = party(F,10,n,t,4, q[4], com)
p6 = party(F,12,n,t,5, q[5], com)

threads = [p1,p2,p3,p4,p5,p6]

#start = time.time()
for t in threads:
    t.start()


for t in threads:
    t.join()

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

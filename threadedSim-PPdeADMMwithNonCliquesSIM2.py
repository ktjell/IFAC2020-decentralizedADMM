# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 14:19:26 2019

@author: kst
"""

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
import ipcon1 as graph
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import networkx as nx
from numba import jit


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
        else:
            print('KØ ER FULD!')
    
class party(Thread):
    party_addr = ips.party_addr
    def __init__(self, F, x, n, t, i, q, com, E, Q, f, retur, g, ite, neighborhood, cliques, vircliques, weights):
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
        self.g = g #generator of field
        self.ite = ite
        self.neighborhood = neighborhood
        self.cliques = cliques
        self.vircliques = vircliques
        self.weights = weights
        self.comr = 0        
        self.comtime = 0
        self.recv = {}
        
    def readQueue(self):
        
        while not self.q.empty():
            b = self.q.get()
            self.recv[b[0]] = b[1]
#            self.q2.put([b[0][-1], b[1]])
            
#    def broadcast(self, name, s):
#        for i in nw.N[self.i]:
#            self.com.com(self.party_addr[i], [name + str(self.i), s])
                    
    
    def get_val(self, name):
        roundcounter = 0
        while name not in self.recv:
            self.readQueue()
            time.sleep(1)
            roundcounter +=1
            if roundcounter > 15:
                print('FAIL', self.i, name)
                return 'FAIL'
        res = self.recv[name]
        del self.recv[name]
        return res
        

    def calVirCliqSum(self,val):
        #ELGAMAL: Every one execpt the 'link' choses random number and send g to the power of it to the other guy
        rliste = []
        for i in self.vircliques[self.i]:
            if i[1] !=  self.i:
                r1 = [self.F.random_element() for j in range(len(val))]
                rliste.append(r1)
                r2 = [pow(self.g,int(str(j)), self.F.p) for j in r1] 
#                print(self.i, 'rand'+ str(i[0]) + str(self.i) )
                self.com.com(self.party_addr[i[1]], ['rand'+ str(i[0]) + str(self.i), r2])
                
        for i in self.vircliques[self.i]:
            if i[1] ==  self.i: #The link supports the comu between the two guys
                for k in i[2:]:
#                    print(self.i, 'rand' + str(i[0]) + str(k))
                    recc = self.get_val('rand' + str(i[0]) + str(k))
                    if type(recc) == str:
                        return 'FAIL'
                    indi = list(set(i) - set([i[0], i[1], k]))[0]
                    self.com.com(self.party_addr[indi], ['rand'+str(i[0])+ str(k), recc])
#        
        #The guyes receives values and creates the keys known only to the two guys
        keys = []
        y=0
        for i in self.vircliques[self.i]:
            if i[1] !=  self.i:
                indi = list(set(i) - set([i[0], i[1], self.i]))[0]
                rec1 = self.get_val('rand' + str(i[0]) + str(indi))
                if type(rec1) == str:
                    return 'FAIL'
                
                keys.append([pow(int(str(rec1[k])), int(str(rliste[y][k])), self.F.p) for k in range(len(val))])
                y+=1

        #the guys chosses random values where there sum = 0, encrypts one of them and sends to other guy
        y=0
        rnlist = []
        for i in self.vircliques[self.i]:
            if i[1] !=  self.i:
                r = np.array([self.F.random_element() for i in range(len(val))])
                rnlist.append(r)
                rr= -r
                
                kr = [keys[y][j] * rr[j] for j in range(len(val))]

#                
                self.com.com(self.party_addr[i[1]], ['rr'+ str(i[0]) + str(self.i), kr])
                y+=1
                
                
        for i in self.vircliques[self.i]:
            if i[1] ==  self.i:
                #The link supports the com
                for k in i[2:]:
                    recc = self.get_val('rr' + str(i[0]) + str(k))
                    if type(recc) == str:
                        return 'FAIL'
                    indi = list(set(i) - set([i[0], i[1], k]))[0]
#                    if self.i ==0:
#                        print('com',k, indi, recc)
                    self.com.com(self.party_addr[indi], ['rr'+str(i[0])+ str(k), recc])
        
#        The guys recieves the values and decryptes and sends their masked val to the link.
        rrand = []
        r5 = []
        y=0
        RES = np.array([self.F(0) for i in range(len(val))])
        for i in self.vircliques[self.i]:
            if i[1] !=  self.i:
                indi = list(set(i) - set([i[0], i[1], self.i]))[0]
                recc = self.get_val('rr' + str(i[0]) + str(indi))
                if type(recc) == str:
                        return 'FAIL'
                rrand = [recc[k]/keys[y][k] for k in range(len(val))]
#                if i[0] == 0:
#                    if self.i ==4 or self.i == 5:
#                        print('rrand', self.i,rrand)
                give = np.array(val/self.weights[self.i][i[1]], dtype = np.int)
#                if self.i == 4:
#                    print('give',give)
                self.com.com(self.party_addr[i[1]], ['val' + str(i[0]) + str(self.i), np.array(rrand) + rnlist[y] + give]) 
                y+=1
#        
            else: #The link recieves the masked values and calculates the result
                for k in i[2:]:
                    
                    aaa = self.get_val('val' + str(i[0]) + str(k))
                    if type(aaa) == str:
                        return 'FAIL'
#                    if self.i == 0:
#                        print('aaa',aaa)
                    r5.append(aaa)
                    
                RES =sum(r5)
#                if self.i==0:
#                    print('RES', RES)
        return RES
    
    def calCliqSum(self,val):
        
        #The guys chooses random numbers the equals 0 end sends one to each guy in clique
        for i in self.cliques[self.i]:
            r1 = []
            for j in range(len(i)-2):
                r1.append([self.F.random_element() for k in range(len(val))])
            r2=np.array(r1)
            r1.append(list(-sum(r2)))   
            
#            print('IN', self.i, 'r'+str(i[0]) + str(self.i))
            for k,j in enumerate(i[1:]):
                self.com.com(self.party_addr[j], ['r'+str(i[0]) + str(self.i), r1[k]])
        

        rec = []
        
#        if self.i==10:
#            self.readQueue()
#            print(self.recv)
        
        for i in self.cliques[self.i]:
            res1 = []
            
            for j in i[1:]:  
                vv = self.get_val('r'+str(i[0])+str(j))
                if type(vv) == str:
                    return 'FAIL'
                res1.append(vv)

            rec.append(sum(np.array(res1)))
        


        for i,ii in enumerate(self.cliques[self.i]):
            for k,j in enumerate(ii[1:]):
                m = np.array(val/self.weights[self.i][j], dtype=np.int) + np.array(rec[i])
                self.com.com(self.party_addr[j],
                               ['val' +str(ii[0]) + str(self.i), m ])

        res = []
        for i in self.cliques[self.i]:
            res1 = []
            for j in i[1:]: 
                vvv = self.get_val('val'+str(i[0])+str(j))
                if type(vvv) == str:
                    return 'FAIL'
                res1.append(vvv)
                
            res.append((np.array(res1)))

        return res
    
        
    def run(self):
        
        scale = 1000
#        print('starting party ', self.i)
        c = 1
        x = 0
        v = np.zeros(2, dtype=np.int)
        v1 = np.zeros(2, dtype=np.int)
        p = np.zeros((2,1))
        ln = len(self.neighborhood[self.i])
        xx = []

        for j in range(self.ite):
#            if self.i == 0:
#                print('Iteration {}.'.format(j))
#            print(self.i, 'v1', v1)
## DISTRIBUTE INPUT
        ## DISTRIBUTE INPUT            
            res = self.calCliqSum(self.x)
            if type(res) == str:
                self.retur.put('FAIL')
            else: 
                self.retur.put(1)

#            s= [sum(i) for i in res]  # sum contributions from each agent in each clique
##            
##            
#            sss1 = sum(s) # sum contribution from each clique
#            if type(sss1) == int:
#                sss1= np.array([0,0])
##                
           
            resVIR = self.calVirCliqSum(self.x) #contribution from each virtual clique
            if type(resVIR) == str:
                self.retur.put('FAIL')
            else: 
                self.retur.put(1)
#            print(self.i,resVIR)
#            print(self.i, sss1, resVIR)
#            sss = sss1 + resVIR #[self.F(sss1[i]) + self.F(resVIR[i]) for i in range(len(sss1))]
            
       
#            print('calculated sum for agent {}: {}'.format(self.i, sum(s)))
# GET OTHER AGENTS DATA
#            
#            ss = []
#            for i in sss:
#                if int(str(i)) > F.p/2:
#                    ss.append((int(str(i)) - F.p) / scale)
#                else:
#                    ss.append(int(str(i))/scale)
##            print(self.i, ss)
#            sm =(ln*v - ss).reshape((2,1))            #sum([v - i for i in ss]) 
#            sp =(ln*v + ss).reshape((2,1))                 #sum([v + i for i in ss]) 
#
##            
#            self.comr+=1
##            s = np.sum(ss)
##            print('Party {} done with com in ite {}.'.format(self.i, j))
#
### START MINIMIZATION
#            p = p + c*sm#( ln* v - s)
#            g = lambda x: self.f(x) + (c/(4*ln)) * np.linalg.norm(  
#                (1./c) * (np.dot(self.E, x) - (1./self.n)*self.Q) - (1/c) * p + sp    # ln*v + s
#                ,2)**2
#            res = minimize(g, x0=0)
#            x = res['x'][0]
#            xx.append(x)
#
#            v = (1./(2*ln)) * (sp #(ln*v+s)
#                     -(1./c)*p 
#                    + (1./c)* (np.dot(self.E, x)- (1./self.n)*self.Q)
#                    )
#            v = v.reshape((2,))
#            vv = [int(i*scale) for i in v]
#            v1 = np.array(vv)
 

    
        
#        print('x:',x)

def findG(p):
    required_set = {num for num in range(1, p)}
    for g in range(1,p):
        if required_set == {pow(g, powers, p) for powers in range(1, p)}:
            return g

mm= 7979490791   
F = field.GF(mm)            
n = 80
NN = [40, 60, 80, 100, 200]

t = 1
m = 2 #number of constraints
g = 109#findG(mm)
ite = 1
#np.random.seed()
ITE = len(NN)
LINKS = 80
links_ar = [n*i for i in range(6,14)]
f = []
for k in range(n):
    f.append(lambda x: ((x-2)**2))

E = []
for k in range(n):
    E.append(np.random.randint(10, size = m).reshape((2,1)))
    
Q = np.random.randint(10,size=m).reshape((2,1))    



retur = que.Queue()
 


timeMAX = []
timeMIN = []
tarray= []
timetemp_array = []
for ii in range(ITE):
    n = NN[ii]
    q = [que.Queue() for i in range(n)]
    com = communicationSimulation(q)
#    LINKS +=10
    timetemp = []
    for i in range(10):
        neighbors, cliques, vircliques, w, edges, links = graph.get_graph_info(n,13*n)
        
        threads = []
        for k in range(n):
            
            threads.append(party(F,np.array([np.random.randint(100), np.random.randint(100)]),n,t,k,q[k],com,E, Q, f, retur, g, ite, neighbors, cliques, vircliques, w))
        
        start = time.time()
            
        for t in threads:
            t.start()
        
        
        for t in threads:
            t.join()   
            
        end = time.time()
        ex = end-start
        
        failtjeck = []
        for i in range(n):
            failtjeck.append(retur.get())
#        print(failtjeck)
        if 'FAIL' in failtjeck:
            print('\n GOT OUT! \n ')
        
        else:
            timetemp.append(ex)
    
    #    print('Execution time: ', np.mean(timetemp))
        print('({},{})'.format(n, ex))
    tarray.append(np.mean(timetemp))
    timeMAX.append(np.max(timetemp))
    timeMIN.append(np.min(timetemp))   
    timetemp_array.append(timetemp)

    
#plt.figure('1')

plt.figure()
plt.plot(NN, tarray)
plt.plot(NN, timeMAX)
plt.plot(NN, timeMIN)


#print(p1.comtime)
#print(p2.comtime)
#print(p3.comtime)
#print('\n')
#print(p1.comtime/ex)
#print(p2.comtime/ex)
#print(p3.comtime/ex)

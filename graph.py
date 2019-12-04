# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 10:27:06 2019

@author: kst
"""

from networkx.generators.random_graphs import erdos_renyi_graph
 
 
n = 6
p = 0.5
g = erdos_renyi_graph(n, p)
 
print(g.nodes)
# [0, 1, 2, 3, 4, 5]
 
print(g.edges)
# [(0, 1), (0, 2), (0, 4), (1, 2), (1, 5), (3, 4), (4, 5)]
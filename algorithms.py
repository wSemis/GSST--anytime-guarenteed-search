import networkx as nx
import numpy as np
import time

def GSST_L(g: "Graph", wall_time: float = 10):
    sta_time = time.time()
    
    while True:
        if time.time() - sta_time > wall_time:
            break
        
        spanning_tree, edges = g.random_spanning_tree()
        spanning_tree.label()
        
        mu = list(spanning_tree.g[0].values())[0]['label']
        
        
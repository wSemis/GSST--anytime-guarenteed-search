num_of_nodes = 9
node_names = [i + 1 for i in range(num_of_nodes)]
edges = [(1,8), (1,4), (2,8), (2,9), (3,4), (4,5), (4,6), (4,9), (5,7), (6,7),]
locations = {
    1: (3, 6),
    2: (5.5, 8),
    3: (6.5, 6.4),
    4: (5, 5),
    5: (4, 3.7),
    6: (6.5,3.4),
    7: (5.2, 2.1),
    8: (3.2, 7.8),
    9: (4.6, 6.8)
}

import sys
sys.path.append('./')
from graph import Graph

fn = 'demo/room'
import os
if not os.path.exists('demo'):
    os.mkdir('demo')
    
if not os.path.exists(fn):
    os.mkdir(fn)

g = Graph(edges, pos=locations)
g.node_size = 1000
g.visualize(save=True,filename=fn+'_graph.png')

from algorithms import GSST_L
g.add_sta(sta=1)
g.pos['sta'] = (1, 6)
g.offset = 0.4
g.generate_random_spanning_tree()
g.visualize(save=True,filename=fn+'_tree.png')

gsst_l = GSST_L(graph=g, filename=fn+'/run')
gsst_l.search(visualize=True)
print(f'COMPLETED!\nTime: {gsst_l.t}, Number of searchers: {gsst_l.num_searcher}, Number of guards: {gsst_l.number_of_guards}')
gsst_l.visualize()

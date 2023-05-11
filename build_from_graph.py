from graphs.gallery_of_art import locations, edges, bg, fig_size

from Graph import Graph
from algorithms import GSST_L

fn = 'demo/art_gallery'
import os
if not os.path.exists('demo'):
    os.mkdir('demo')
if not os.path.exists(fn+'/'):
    os.mkdir(fn+'/')
        
print(max([i[0] for i in locations.values()]))
print(max([i[1] for i in locations.values()]))
g = Graph(edges, pos=locations)
g.bg = bg
g.fig_size = fig_size
g.node_size = 1000

g.add_sta(sta=1)
g.pos['sta'] = (14, 6)
g.visualize(save=True,filename=fn+'_graph.png')

g.generate_random_spanning_tree()
g.offset = 0.4
g.visualize(save=True,filename=fn+'_tree.png')

gsst_l = GSST_L(graph=g, filename=fn+'/run')
gsst_l.search(visualize=True)
print(f'COMPLETED!\nTime: {gsst_l.t}, Number of searchers: {gsst_l.num_searcher}, Number of guards: {gsst_l.number_of_guards}')
gsst_l.visualize()

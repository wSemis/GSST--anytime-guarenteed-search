from Graph import Graph
from GSST import GSST
import os
import pickle

fn = 'gif/{}'

def main(idx=0):
    if not os.path.exists('gif'):
        os.mkdir('gif')
        
    G = Graph()
    G.add_sta()
    G.generate_random_spanning_tree()
    G.t.label()
    G.t.visualize(save=True, filename=fn.format(f'{idx}_tree.png'))
    pickle.dump(G, open(f'gif/{idx}_graph.pkl', 'wb'))
    
    gsst = GSST(tree=G.t, filename=fn.format(idx))
    gsst.search(visualize=True)
    print(f'COMPLETED!\nTime: {gsst.t}, Number of searchers: {gsst.num_searcher}')
    gsst.visualize()    

if __name__ == '__main__':
    for i in range(10):
        main(i)
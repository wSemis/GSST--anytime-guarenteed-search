from graph import Graph
from gsst import GSST, GSST_L, GSST_R
import os
import pickle

fn = 'gif/{}'

def test_trees(rep=10):
    if not os.path.exists('gif'):
        os.mkdir('gif')
    for idx in range(rep):    
        G = Graph()
        G.add_sta()
        G.generate_random_spanning_tree()
        G.t.visualize(save=True, filename=fn.format(f'{idx}_tree.png'))
        pickle.dump(G, open(f'gif/{idx}_graph.pkl', 'wb'))
        gsst = GSST(tree=G.t, filename=fn.format(idx))
        gsst.search(visualize=True)
        print(f'COMPLETED!\nTime: {gsst.t}, Number of searchers: {gsst.num_searcher}')
        gsst.visualize()    

def test_GSST_L(rep=10):
    if not os.path.exists('gif'):
        os.mkdir('gif')
    for idx in range(rep):
        G = Graph()
        G.add_sta()
        G.generate_random_spanning_tree()
        G.visualize(save=True, filename=fn.format(f'{idx}_graph.png'))
        pickle.dump(G, open(f'gif/{idx}_graph.pkl', 'wb'))
        gsst_l = GSST_L(graph=G, filename=fn.format(idx))
        gsst_l.search(visualize=True)
        print(f'COMPLETED!\nTime: {gsst_l.t}, Number of searchers: {gsst_l.num_searcher}, Number of guards: {gsst_l.number_of_guards}')
        gsst_l.visualize()

def test_GSST_R(rep=10):
    if not os.path.exists('gif'):
        os.mkdir('gif')
    for idx in range(rep):    
        G = Graph()
        G.add_sta()
        G.generate_random_spanning_tree()
        G.visualize(save=True, filename=fn.format(f'{idx}_graph.png'))
        pickle.dump(G, open(f'gif/{idx}_graph.pkl', 'wb'))
        gsst_r = GSST_R(graph=G, filename=fn.format(idx))
        gsst_r.search(visualize=False)

def main():
    # test_trees(10)
    # test_GSST_L(50)
    test_GSST_R(1)

if __name__ == '__main__':
    main()

import numpy as np
import networkx as nx
from copy import deepcopy
from Graph import Graph
import os
import imageio

WALL_TIME = 5
class GSST:
    def __init__(self, graph: "Graph"=None, tree: "Graph"=None, filename='test_run') -> None:
        if tree == None:
            self.graph = graph
            self.spanning_tree, self.B = self.graph.get_spanning_tree()
        elif graph == None:
            self.graph = tree
            if self.graph.g.is_directed:
                self.graph.g = self.graph.g.to_undirected()
            self.spanning_tree, self.B = tree, []
        elif graph == None and tree == None:
            raise ValueError("Either graph or tree should not be None")
        
        self.num_searcher = self.spanning_tree.mu
        self.t = 0
        self.N = self.graph.g.number_of_nodes()
        
        #Visitation status
        self.to_visit = {i for i in self.graph.g.nodes()}
        self.to_visit.remove('sta')
        self.visited = {i: False for i in self.graph.g.nodes()}
        self.visited['sta'] = True
        self.unvisited_g = {n: self.graph.g.degree[n]
            for n in self.graph.g.nodes()}
        self.unvisited_g[graph.start] -= 1
        self.unvisited_t = {n: self.spanning_tree.g.out_degree[n]
            for n in self.spanning_tree.g.nodes()}
        self.unvisited_t[graph.start] -= 1
        
        #Searcher locations
        self.searcher_locations = ['sta' for _ in range(self.num_searcher)]
        self.searcher_per_locations ={i: 0 for i in self.graph.g.nodes()}
        self.searcher_per_locations['sta'] = self.num_searcher
        
        self.set_node_attributes()
        self.history = []
        self.save_history()
        
        self.fn = filename
    
    def set_node_attributes(self) -> None:
        nx.set_node_attributes(self.graph.g, self.searcher_per_locations, 'searcher_number')
        nx.set_node_attributes(self.graph.g, self.visited, 'visited')
        
    ## Might not be correct, think about labels?
    def can_move_searcher(self, node) -> bool:
            if self.searcher_per_locations[node] > 1:
                return True
            
            if self.searcher_per_locations[node] == 0:
                raise ValueError("No searcher at node {}".format(node))
                                
            # print(f'Node {node} has {unvisited} unvisited neighbors among {self.spanning_tree.g[node]}')
            return self.unvisited_t[node] <= 1
    
    def searcher_to_new_node(self, node) -> None:
        self.visited[node] = True
        for n in self.graph.g[node]:
            self.unvisited_g[n] -= 1
        for n in self.spanning_tree.g[node]:
            self.unvisited_t[n] -= 1
    
    def move_searcher(self, num, node, positive_edge=True) -> None:
        prev_node = self.searcher_locations[num]
        self.searcher_per_locations[prev_node] -= 1
        
        self.searcher_locations[num] = node
        self.searcher_per_locations[node] += 1
        
        if self.visited[node] == False:
            self.searcher_to_new_node(node)
            
        if node in self.to_visit:
            self.to_visit.remove(node)
        
        # print(f'Searcher {num} moves from {prev_node} to {node}')
        if positive_edge:
            self.spanning_tree.g[prev_node][node]['label'] -= 1
        else:
            self.spanning_tree.g[prev_node][node]['label'] += 1

    
    def save_history(self) -> None:
        self.history.append(deepcopy(self.graph))
    
    def search_step(self) -> None:
        for i in range(self.num_searcher):
            node = self.searcher_locations[i]
            can_move = self.can_move_searcher(node)
            
            # print(f'Searcher {i} at {node} & t={self.t} can move: {can_move}')
            if not can_move:
                continue
            
            adj = list(self.spanning_tree.g[node])
            edge_labels = np.array([self.spanning_tree.g.edges[(node, neighbor)]['label'] for neighbor in adj])      
            positive = np.where(edge_labels > 0)[0]
            negative = np.where(edge_labels < 0)[0]
            
            if len(positive) > 0:
                idx = np.argmin(edge_labels[positive])
                # print(edge_labels)
                # print(positive)
                # print(positive[idx])
                # print(adj)
                next_node = adj[positive[idx]]
                self.move_searcher(i, next_node)
            elif len(negative) > 0:
                next_node_idx = negative[0]
                next_node = adj[next_node_idx]
                self.move_searcher(i, next_node, positive_edge=False)
            else:
                continue
            
            self.after_search_step()
            
    def after_search_step(self) -> None:
        pass
        
    def search(self, visualize=False) -> None:
        print('Search started with {} searchers'.format(self.num_searcher))
        
        self.png_saved = visualize
        if visualize:
            self.history[-1].visualize(save=True, filename=f'{self.fn}_{self.t}.png')
                                
        while len(self.to_visit) != 0:
            # print(f'At time {self.t}, {(self.to_visit)} nodes left to visit')
            if self.t > WALL_TIME * self.N:
                print(f'INTERRUPTED!\nTime: {self.t}, Number of searchers: {self.num_searcher}, unvisited area: {self.to_visit}')
                exit()
                
            self.search_step()
                
            self.set_node_attributes()      
            self.save_history()
            self.t += 1

            if visualize:
                self.visualize_step(self.t)
    
    def visualize(self) -> None:
        fns = []
        for i in range(self.t + 1):
            if not self.png_saved:
                self.visualize_step(i)
            fns.append(f'{self.fn}_{i}.png')
        
        imageio.mimsave(f'{self.fn}.gif', [imageio.imread(filename) for filename in fns], duration=1)
                
        for filename in set(fns):
            # os.remove(filename)
            pass
        
    def visualize_step(self, step: int) -> None:
        self.history[step].visualize(save=True, filename=f'{self.fn}_{step}.png')
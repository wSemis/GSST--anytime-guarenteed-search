import numpy as np
from copy import deepcopy
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
BRANCHING_FACTOR = 4

class Graph:
    def __init__(self, arg= None, directed=False) -> None:
        self.is_directed = directed
        
        if arg is None:
            edgeList = self.random_graph()
            edge_attrs = -1
        elif isinstance(arg, nx.Graph):
            edgeList = arg.edges()
            edge_attrs = nx.get_edge_attributes(arg, 'label')
            print('edge attr',edge_attrs)
        elif isinstance(arg, list):
            edgeList = arg
            edge_attrs = -1
        else:
            raise NotImplementedError
        
        if directed:
            self.g = nx.DiGraph(edgeList)
        else:
            self.g = nx.Graph(edgeList)
        
        nx.set_edge_attributes(self.g, edge_attrs, 'label')
    
    def add_sta(self, sta=0):
        self.g.add_edge(*('sta', sta))
    
    def random_graph(self) -> list[list[int]]:
        k = BRANCHING_FACTOR
        print(f'Randomed branching factor k = {k}')
        N = np.random.randint(10, 20)

        edgeList = []
        visited = [0]
        to_visit = [i for i in range(1, N)]
        
        for _ in range(N-1):
            sta = np.random.choice(visited)
            end = np.random.choice(to_visit)
            to_visit.remove(end)
            visited.append(end)
            a,b = min(sta, end), max(sta, end)
            edgeList.append((a,b))
                
        edges = {}
        for i in range(N):
            for j in range(i+1, N):
                edges[(i, j)] = 1
            
        for edge in edgeList:
            edges.pop(edge)
        
        edges = list(edges.keys())
        length = k * N // 2 - (N-1)
        idxs = np.random.choice(len(edges), length, replace=False)
        for i in idxs:
            edge = edges[i]
            edgeList.append(edge)
        return edgeList
    
    def is_tree(self):
        if self.g.is_directed():
            return nx.is_tree(self.g.to_undirected())
        return nx.is_tree(self.g)
    
    def generate_random_spanning_tree(self):
        if self.is_tree():
            return self
        else:
            # return Graph(nx.random_spanning_tree(self, weight=None))
            
            # Use Algorithm 4
            edges = []
            sta = 'sta'
            visited = {node: False for node in self.g.nodes()}
            parents = dict()
            
            while len(edges) != self.g.number_of_nodes() - 1:
                visited[sta] = True
                neighbors = []
                
                for v in self.g[sta]:
                    if not visited[v]:
                        neighbors.append(v)
                
                if len(neighbors) == 0:
                    sta = parents[sta]
                    continue
            
                end = np.random.choice(neighbors)
                parents[end] = sta
                edges.append((sta, end))
                
                sta = end
            
            non_tree_edges = []
            for a,b in self.g.edges():
                if (a,b) in edges or (b,a) in edges:
                    continue
                non_tree_edges.append((min(a,b), max(a,b)))
                    
            self.t = Graph(edges, directed=False)
            self.B = non_tree_edges
            self.t.label()
           
    def get_spanning_tree(self):
        if not hasattr(self, 't'):
            self.generate_random_spanning_tree()
        return self.t, self.B
    
    def label_reverse(self, parents: list[int]):
        for i in range(0, self.g.number_of_nodes()):
            if i in parents:
                self.g.edges[(i, parents[i])]['label'] = - self.g.edges[(parents[i], i)]['label']
    
    def label(self):
        assert self.is_tree(), "Method only applicable to trees"

        parents = dict(nx.bfs_predecessors(self.g, 'sta'))

        def is_leaf(node):
            if self.is_directed:
                # Leaf as in an undirected graph
                return self.g.in_degree(node) == 1 and node != 'sta'
            else:
                return self.g.degree(node) == 1 and node != 'sta'
        
        def get_edge_info(node):
            adj = list(self.g[node])
            edge_labels = [self.g.edges[(node, neighbor)]['label'] for neighbor in adj]      
            label_counts = Counter(edge_labels)
            return adj, edge_labels, label_counts
        
        def get_parent(node):
            return parents[node]
        
        buffer = [x for x in self.g.nodes() 
                  if is_leaf(x)]
        while len(buffer) != 0:
            node = buffer.pop(0)
            
            if is_leaf(node):
                parent = get_parent(node)
                self.g.edges[(parent, node)]['label'] = 1
                
            else:
                adj, edge_labels, label_counts = get_edge_info(node)
                
                if label_counts[-1] == 1:
                    child = adj[edge_labels.index(-1)]
                    l_max = max(edge_labels)
                    if l_max == -1: continue
                    max_cout = label_counts[l_max]
                    if max_cout == 1:
                        self.g.edges[(node, child)]['label'] = l_max
                    else:
                        self.g.edges[(node, child)]['label'] = l_max + 1
              
            if node != 'sta':
                parent = get_parent(node)
                adj, edge_labels, label_counts = get_edge_info(parent)
                if label_counts[-1] == 1:
                    buffer.append(parent)
        
        # Label root
        # adj, edge_labels, label_counts = get_edge_info(0)
        # max_edge = max(edge_labels)
        # if label_counts[max_edge] == 1:
        #     self.mu = max_edge
        # else:
        #     self.mu = max_edge + 1
        self.mu = self.g.edges[('sta', 0)]['label']
        self.g = self.g.to_directed()
        self.label_reverse(parents)
        
    def visualize(self, save=False, filename='testrun', ax=None):
        if ax is None:
            fig, ax = plt.subplots(1,1, figsize=(10, 10))
        
        if not self.is_tree():
            if not hasattr(self, 't'):
                pos = nx.spring_layout(self.g, k=3)
                nx.draw(self.g, pos=pos, with_labels=True, node_color='c', ax=ax)
                # nx.draw_networkx_edge_labels(self.g, pos=pos, edge_labels=nx.get_edge_attributes(self.g, 'label'), label_pos=0.6, ax=ax)
            else:
                pos = nx.spring_layout(self.g, k=3)
                try:
                    visited_nodes = {node for node in self.g.nodes() if self.g.nodes[node]['visited']}
                except KeyError:
                    visited_nodes = set()
                colors = ['g' if node in visited_nodes else 'c' for node in self.g.nodes()]
                nx.draw_networkx_nodes(self.g, pos=pos, node_color=colors, ax=ax)
                nx.draw_networkx_labels(self.g, pos=pos, ax=ax)

                tree_edges = self.t.g.edges()
                non_tree_edges = self.B
                nx.draw_networkx_edges(self.g, pos=pos, edgelist=tree_edges, edge_color='r', ax=ax)
                nx.draw_networkx_edges(self.g, pos=pos, edgelist=non_tree_edges, style='dashed', ax=ax)
        else:
            if self.g.is_directed():
                pos = nx.nx_agraph.graphviz_layout(self.g, prog='dot',args="-Grankdir=LR")
                try:
                    visited_nodes = {node for node in self.g.nodes() if self.g.nodes[node]['visited']}
                except KeyError:
                    visited_nodes = set()
                colors = ['g' if node in visited_nodes else 'c' for node in self.g.nodes()]
                nx.draw_networkx_nodes(self.g, pos=pos, node_color=colors, node_size=300, ax=ax)
                nx.draw_networkx_labels(self.g, pos=pos, ax=ax)
                nx.draw_networkx_labels(self.g, pos={k: (x, y + 4) for k,(x,y) in pos.items()}, labels=nx.get_node_attributes(self.g, 'searcher_number'), font_color='r', ax=ax)
                
                G = self.g
                curved_edges = [edge for edge in G.edges() if reversed(edge) in G.edges()]
                straight_edges = list(set(G.edges()) - set(curved_edges))
                nx.draw_networkx_edges(G, pos, ax=ax)
                # arc_rad = 0.15
                # nx.draw_networkx_edges(G, pos, edgelist=curved_edges, connectionstyle=f'arc3, rad = {arc_rad}', ax=ax)
                # nx.draw_networkx_edge_labels(self.g, pos=pos, edge_labels=nx.get_edge_attributes(self.g, 'label'), label_pos=0.6. ax=ax)
                
            else:
                pos = nx.nx_agraph.graphviz_layout(self.g, prog='dot')
                nx.draw(self.g, pos=pos, with_labels=True, node_color='c', ax=ax)
                # nx.draw_networkx_edge_labels(self.g, pos=pos, edge_labels=nx.get_edge_attributes(self.g, 'label'), label_pos=0.6)
        
        if save:
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()
            
    
    
if __name__ == "__main__":
    g = Graph()
    g.add_sta()
    print('Vertex number: ',g.g.number_of_nodes())
    print('Edge number: ', len(g.g.edges()))
    print('Actual Branching Factor', np.average([tup[1] for tup in g.g.degree()]))
    print(g.g[0])
    g.visualize()
    
    print()
    print("Random spanning tree")
    g.generate_random_spanning_tree()
    g.visualize()
    # print(g.t.g[0])
    # print('is tree: ',g.t.is_tree())
    # print('Vertex number: ', len(g.t.g.nodes()))
    # print(f'Tree edges:{len(g.t.g.edges())} \nNon-tree edges: {len(g.B)}')
    # g.t.visualize()
    
    # g.t.label()
    # g.t.visualize()
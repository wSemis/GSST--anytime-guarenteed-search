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
        return nx.is_tree(self.g)
    
    def random_spanning_tree(self):
        if self.is_tree():
            return self
        else:
            # return Graph(nx.random_spanning_tree(self, weight=None))
            
            # Use Algorithm 4
            edges = []
            sta = 0
            visited = np.zeros(self.g.number_of_nodes(), dtype=bool)
            parents = np.zeros(self.g.number_of_nodes(), dtype=int)
            
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
        
    def label(self):
        assert self.is_tree(), "Method only applicable to trees"

        parents = dict(nx.bfs_predecessors(self.g, 0))

        def is_leaf(node):
            if self.is_directed:
                return self.g.in_degree(node) == 1 and self.g.out_degree(node) == 0
            else:
                return self.g.degree(node) == 1 and node != 0
        
        def get_edge_info(node):
            adj = list(self.g[node])
            edge_labels = [self.g.edges[(node, out)]['label'] for out in adj]      
            label_counts = Counter(edge_labels)
            return adj, edge_labels, label_counts
        
        def get_parent(node):
            return parents[node]
        
        buffer = [x for x in self.g.nodes() 
                  if is_leaf(x)]
        while len(buffer) != 0:
            node = buffer.pop()
            
            if is_leaf(node):
                parent = get_parent(node)
                self.g.edges[(parent, node)]['label'] = 1
                
            else:
                adj, edge_labels, label_counts = get_edge_info(node)
                
                if label_counts[-1] == 1:
                    out = adj[edge_labels.index(-1)]
                    l_max = max(edge_labels)
                    if l_max == -1: continue
                    max_cout = label_counts[l_max]
                    if max_cout == 1:
                        self.g.edges[(node, out)]['label'] = l_max
                    else:
                        self.g.edges[(node, out)]['label'] = l_max + 1
              
            if node != 0:
                parent = get_parent(node)
                adj, edge_labels, label_counts = get_edge_info(parent)
                if label_counts[-1] == 1:
                    buffer.append(parent)
                  
    def visualize(self):
        if not self.is_tree():
            pos = nx.spring_layout(self.g, k=3)
            nx.draw(self.g, pos=pos, with_labels=True)
            nx.draw_networkx_edge_labels(self.g, pos=pos, edge_labels=nx.get_edge_attributes(self.g, 'label'), label_pos=0.6)
        else:
            pos = nx.nx_agraph.graphviz_layout(self.g, prog='dot')
            nx.draw(self.g, pos=pos, with_labels=True)
            nx.draw_networkx_edge_labels(self.g, pos=pos, edge_labels=nx.get_edge_attributes(self.g, 'label'), label_pos=0.6)
        plt.show()
            
    
    
if __name__ == "__main__":
    g = Graph()
    print('Vertex number: ',g.g.number_of_nodes())
    print('Edge number: ', len(g.g.edges()))
    print('Actual Branching Factor', np.average([tup[1] for tup in g.g.degree()]))
    print(g.g[0])
    g.visualize()
    
    print()
    print("Random spanning tree")
    g.random_spanning_tree()
    print(g.t.g[0])
    print('is tree: ',g.t.is_tree())
    print('Vertex number: ', len(g.t.g.nodes()))
    print(f'Tree edges:{len(g.t.g.edges())} \nNon-tree edges: {len(g.B)}')
    g.t.visualize()
    
    g.t.label()
    g.t.visualize()
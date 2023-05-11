import networkx as nx
import numpy as np
import time
from GSST import GSST
from Graph import Graph
# from typing import override

class GSST_L(GSST):
    def __init__(self, graph: Graph = None, filename='test_run') -> None:
        self.number_of_guards = 0
        self.N = graph.g.number_of_nodes()
        self.guard_per_locations = {i: 0 for i in graph.g.nodes()}
        self.guard_per_locations['sta'] = self.number_of_guards
        self.to_guard = None

        super().__init__(graph, filename=filename)
        self.guard_locations = []
        self.guard_degree = {0: set(), 1:set()}
        
        self.non_tree_edge_nodes = {i: False for i in graph.g.nodes()}
        self.non_tree_edge_nodes['sta'] = False
        for edge in self.B:
            a, b = edge
            self.non_tree_edge_nodes[a] = True
            self.non_tree_edge_nodes[b] = True
            
    def call_guard(self, node):
        assert node != 'sta' or self.print_guard_info('Shuold not call guard at sta')
        
        guard = None
        if len(self.guard_degree[0]) > 0:
            guard = self.guard_degree[0].pop()
        
        elif self.visited[node] == False:
            for g in self.guard_degree[1]:
                loc = self.guard_locations[g]
                for n in self.graph.g[loc]:
                    if node == n:
                        guard = g
                        break
          
        if guard == None:          
            guard = self.add_guard()

        self.move_guard(guard, node)

    def print_guard_info(self, txt):
        print()
        print('~'*50)
        print(txt)
        print(f'Number of guards: {self.number_of_guards}')
        print(f'Guard locations: {self.guard_locations}')
        print(f'Guard per locations: {self.guard_per_locations}')
        print(f'Guard degree: {self.guard_degree}')
        print('~'*50)
        return False
    
    def add_guard(self) -> None:
        assert self.guard_per_locations['sta'] == 0 or self.print_guard_info('Have existing guards available')
        
        self.guard_locations.append('sta')
        self.guard_per_locations['sta'] += 1
        self.number_of_guards += 1
        return self.number_of_guards - 1
    
    def free_guard(self, guard) -> None:
        prev_loc = self.guard_locations[guard]
        if prev_loc != None:
            self.guard_per_locations[prev_loc] -= 1
    
        self.guard_locations[guard] = 'sta'
        self.guard_per_locations['sta'] += 1
        
    def add_guard_in_degree(self, guard, deg):
        if deg in self.guard_degree:
            self.guard_degree[deg].add(guard)
    
    def check_guard_in_degree(self, guard, deg=None):
        if deg == None:
            return guard in self.guard_degree[0], 0 or guard in self.guard_degree[1], 1
        
        assert deg in [0,1] or self.print_guard_info(f'Wrong degree for {guard, deg}')
        return guard in self.guard_degree[deg], deg
    
    def remove_guard_from_degree(self, guard, deg=None):
        if deg == None:
            if guard in self.guard_degree[0]:
                self.guard_degree[0].remove(guard)
            elif guard in self.guard_degree[1]:
                self.guard_degree[1].remove(guard)
        else:
            if deg in self.guard_degree:
                self.guard_degree[deg].remove(guard)
    
    def move_guard(self, guard, node) -> None:
        self.free_guard(guard)    
        self.remove_guard_from_degree(guard)
    
        deg = self.unvisited_g[node]
        if deg == 0:
            raise ValueError("Guard should not be at a node with no unvisited neighbors")
        self.add_guard_in_degree(guard, deg)

        self.guard_locations[guard] = node
        self.guard_per_locations[node] += 1
        self.guard_per_locations['sta'] -= 1
        
    # @override
    def set_node_attributes(self) -> None:
        super().set_node_attributes()
        nx.set_node_attributes(self.graph.g, self.guard_per_locations, 'guard_number')

    # @override
    def searcher_to_new_node(self, node) -> None:
        super().searcher_to_new_node(node)
        for neighbor in self.graph.g[node]:
            deg = self.unvisited_g[neighbor]
            if deg >= 2: continue
        
            guards_to_update = [g for g, loc in enumerate(self.guard_locations) if loc == neighbor]
            for g in guards_to_update:
                self.remove_guard_from_degree(g)
                self.add_guard_in_degree(g, deg)
                if deg == 0:
                    self.free_guard(g)
            
    def after_search_step(self) -> None:
        if self.to_guard == None:
            return

        node = self.to_guard
        self.to_guard = None
        
        if  self.unvisited_g[node] == 0 or\
            self.guard_per_locations[node] > 0 or\
            self.searcher_per_locations[node] > 0:
            return 
            
        self.call_guard(node)
        
    # @override
    def can_move_searcher(self, node) -> bool:
        tree_can_move = super().can_move_searcher(node)
        
        print(f'Node {node}: tree_can_move:{tree_can_move}')
        print(f'unvisited_g: {self.unvisited_g[node]}, unvisited_t: {self.unvisited_t[node]}')
        print(f'self.guard_per_locations[node]: {self.guard_per_locations[node]}')
        print(f'self.searcher_per_locations[node]: {self.searcher_per_locations[node]}')
        if self.unvisited_g[node] == 0 and node != 'sta':
            assert self.guard_per_locations[node] == 0 or self.print_guard_info(f'Guard at node {node} should have been cleared')
            
        if tree_can_move == False:
            return False
        
        if self.non_tree_edge_nodes[node] == False or \
            self.unvisited_g[node] == 0 or\
            self.guard_per_locations[node] > 0 or\
            self.searcher_per_locations[node] > 1:
            return True
        
        self.to_guard = node
        return True
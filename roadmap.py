from IPython.core.debugger import set_trace

import numpy as np
from math import atan2
import random
from matplotlib import collections as mc

class Roadmap:
    """A class to represent a road network"""

    def __init__(self, nodes, edges, rotate=True, bidirectional=True):
        """
        nodes: list of tuples (x, y). Defines the cartesian location of each intersection.
        edges: list of tuples (start, end). Defines the roads between intersections. Each edge is
            unidirectional.
        """
        if rotate:
            first = 1
            second = 0
        else:
            first = 0
            second = 1
        #
        self.graph = {(node[first],node[second]) : {} for node in nodes}
        for edge in edges:
            a = (nodes[edge[0]][first],nodes[edge[0]][second])
            b = (nodes[edge[1]][first],nodes[edge[1]][second])
            slope = [b[0] - a[0], b[1] - a[1]]
            dist = np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
            self.graph[a][b] = dist

            if bidirectional:
                self.graph[b][a] = dist
            #
        #
        self._total_len = 0.0
        for dests in self.graph.values():
            self._total_len += np.sum(list(dests.values()))
        #

        # ==============================
        # ==============================

        self.nodes_sorted = sorted(self.graph.keys())
        new_ids = []
        edge_sort_tup = []
        self.edges_sorted = []
        for ii in nodes:
            new_ids.append(self.nodes_sorted.index(ii))
        #
        # # # --> new_ids = [0, 2, 4, 1, 3, 5]
        # set_trace()
        for ii in range(len(edges)):
            for jj in range(len(edges[ii])):
                edge_sort_tup.append( new_ids[edges[ii][jj]] )
            #
            next_edge = tuple(edge_sort_tup)
            self.edges_sorted.append(next_edge)
            edge_sort_tup = []
        #
        # set_trace()
        node_ids, neighbor_counts = np.unique( self.edges_sorted, return_counts=True )

        # node_ids, neighbor_counts = np.unique( np.asarray(edges), return_counts=True )
        self.max_neighbors = np.max( neighbor_counts )
        self.select_act = np.full( (len(node_ids),self.max_neighbors), np.nan )

        edg = np.array( self.edges_sorted ) #.astype(int)

        max_neighbor_idx = np.arange(self.max_neighbors)
        mix_locs = max_neighbor_idx

        for ii in node_ids:
            rows = np.where( edg == ii )[0]
            ii_mat = np.unique( edg[rows,:] )
            idx_ii = np.nonzero( ii_mat != ii )
            ell = ii_mat[idx_ii]

            # mix_locs = np.random.shuffle(max_neighbor_idx)
            np.random.shuffle(mix_locs)

            for its, jj in enumerate(ell):
                self.select_act[ii,mix_locs[its]] = jj
            #
        #
        # self.select_act.astype(int)
        # set_trace()
        # set_trace()
        # print(actions)
        # for key,value in self.graph.items():
        #     actions = max(actions, len(value))
    #
    def get_nearest_waypoint(self, pos):
        waypoint = None
        min_dist = 999999999
        for node in self.graph:
            dist = (pos[0] - node[0])**2 + (pos[1] - node[1])**2
            if dist < min_dist:
                min_dist = dist
                waypoint = node
            #
        #
        return waypoint
    #
    def get_next_waypoint(self, waypoint, psi):
        options = self.graph[waypoint].keys()
        next_wp = random.choice(options)
        next_psi = atan2(next_wp[0] - waypoint[0], next_wp[1] - waypoint[1])
        diff_angle = abs(((next_psi - psi) + np.pi) % (2*np.pi) - np.pi)
        return next_wp
    #
    def get_loc(self, state):
        """Returns the (x,y) location of a position along an edge

        edge -- tuple containing the (x,y) coordnates of
                the beginning and ending nodes of the edge
        pos  -- value between 0 and 1 indicating the distance along the edge
        """

        edge = state[0]
        pos = state[1]

        try:
            self.graph[edge[0]][edge[1]]
        except KeyError:
            raise ValueError("Invalid roadmap edge.")
        #
        loc = (pos*edge[1][0] + (1-pos)*edge[0][0],
               pos*edge[1][1] + (1-pos)*edge[0][1])
        return loc
    #
    def get_edge_weight(self, waypoint_a, waypoint_b):
        #TODO add in particle density to weight
        #   (more particles should decrease the weight making it more likely)
        return np.linalg.norm(np.array(waypoint_b) - np.array(waypoint_a))
    #
    def get_new_destination(self, current):
        options = self.graph.keys()
        options.remove(current[:2])
        next = random.choice(options)
        return (next[0], next[1], -100)
    #
    def get_lurd( self, node ):

        neighbors = self.graph[node]

        return lurd
    #
    def djikstra(self, waypoint, destination):
        visited = {waypoint[:2]: 0}
        path = {}
        nodes = set(self.graph.keys())
        while nodes:
            min_node = None
            for node in nodes:
                if node in visited.keys():
                    if min_node is None:
                        min_node = node
                    elif visited[node] < visited[min_node]:
                        min_node = node
                    #
                #
            #
            if min_node is None:
                break
            #
            nodes.remove(min_node)
            current_weight = visited[min_node]

            for edge in self.graph[min_node]:
                weight = current_weight + self.getEdgeWeight(min_node, edge)
                if edge not in visited or weight < visited[edge]:
                    visited[edge] = weight
                    path[edge] = min_node
                #
            #
        #
        shortest_path = [path[destination[0:2]], destination[0:2]]
        while shortest_path[0] != waypoint[:2]:
            shortest_path.insert(0,path[shortest_path[0]])
        #
        return shortest_path
    #
    @property
    def total_length(self):
        return self._total_len
    #
    def get_action_dimensions(self):
        # actions =  len(self.graph.keys())
        actions =  self.max_neighbors
        # print(actions)
        # for key,value in self.graph.items():
        #     actions = max(actions, len(value))
        return actions
    #
    # def visualize(self, ax):
    #     edges = []
    #     for a in self.graph:
    #         edges += [(a, b) for b in self.graph[a]]
    #     lc = mc.LineCollection(edges, colors=[(0,0,0,1)])
    #     ax.add_collection(lc)
    #
    def getMinMaxXY(self):
        initial_node = list(self.graph.keys())[0]
        print(initial_node)
        min_x = initial_node[0]
        max_x = initial_node[0]
        min_y = initial_node[1]
        max_y = initial_node[1]
        for a in self.graph.keys():
            if a[0] < min_x: min_x = a[0]
            if a[0] > max_x: max_x = a[0]
            if a[1] < min_y: min_y = a[1]
            if a[1] > max_y: max_y = a[1]
            for b in self.graph[a].keys():
                if b[0] < min_x: min_x = b[0]
                if b[0] > max_x: max_x = b[0]
                if b[1] < min_y: min_y = b[1]
                if b[1] > max_y: max_y = b[1]
        return min_x, max_x, min_y, max_y

    def visualize(self, ax):
        edges = []
        for a in self.graph:
            edges += [(a, b) for b in self.graph[a]]
        #
        lc = mc.LineCollection(edges, colors=[(0,0,0,1)])
        ax.add_collection(lc)
    #
#

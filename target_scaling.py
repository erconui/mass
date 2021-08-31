from __future__ import division

import time
import random
import numpy as np
from scipy.stats import norm
import scipy.io as sio
import matplotlib as mpl
from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import collections as mc
from scipy.stats import multivariate_normal as mvn
from math import atan2
from tqdm import tqdm, trange
import copy
import datetime
from multiprocessing import Pool

class Roadmap:
    """A class to represent a road network"""

    def __init__(self, nodes, edges, bidirectional=True):
        """
        nodes: list of tuples (x, y). Defines the cartesian location of each intersection.
        edges: list of tuples (start, end). Defines the roads between intersections. Each edge is
            unidirectional.
        """
        self.graph = {node : {} for node in nodes}
        for edge in edges:
            a = nodes[edge[0]]
            b = nodes[edge[1]]
            dist = np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
            self.graph[a][b] = dist
            if bidirectional:
                self.graph[b][a] = dist

        self._total_len = 0.0
        for dests in self.graph.values():
            self._total_len += np.sum(list(dests.values()))

    def get_nearest_waypoint(self, pos):
        waypoint = None
        min_dist = 999999999
        for node in self.graph:
            dist = (pos[0] - node[0])**2 + (pos[1] - node[1])**2
            if dist < min_dist:
                min_dist = dist
                waypoint = node
        return waypoint

    def get_next_waypoint(self, waypoint, psi):
        options = self.graph[waypoint].keys()
        next_wp = random.choice(options)
        next_psi = atan2(next_wp[0] - waypoint[0], next_wp[1] - waypoint[1])
        diff_angle = abs(((next_psi - psi) + np.pi) % (2*np.pi) - np.pi)
        return next_wp

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

        loc = (pos*edge[1][0] + (1-pos)*edge[0][0],
               pos*edge[1][1] + (1-pos)*edge[0][1])
        return loc
#         raise NotImplementedError

    @property
    def total_length(self):
        return self._total_len

    def visualize(self, ax):
        edges = []
        for a in self.graph:
            edges += [(a, b) for b in self.graph[a]]
        lc = mc.LineCollection(edges, colors=[(0,0,0,1)])
        ax.add_collection(lc)

    def edge_list(self):
        edges = []
        lengths = []
        for start_node in self.graph:
            for end_node in self.graph[start_node]:
                if not((start_node, end_node) in edges or (end_node, start_node) in edges):
                    edges.append((start_node, end_node))
                    lengths.append(self.graph[start_node][end_node])
        return edges, lengths/sum(lengths)

class Particle(object):
    def __init__(self, roadmap, v0, dt, e0=None, x0=None, sigma=0.1, name=None):
        """A Particle contains the state and dynamic model of one hypothesis of a vehicle location.

        The particle's state consists of which road segment the vehicle is on and how far along
        that road segment the vehicle is, on a scale of 0 to 1. The particle also stores its
        nominal velocity and noise characteristics.

        roadmap -- object containing a graph describing the network of roads

        """
        # current edge
        self._roadmap = roadmap
#         print(roadmap.edge_list())
        if e0 is None:
#             a = random.choice(list(self._roadmap.graph.keys()))
#             b = random.choice(list(self._roadmap.graph[a].keys()))
#             self._e = (a, b)
            options, probabilities = roadmap.edge_list()
            self._e = options[np.random.choice(range(len(options)), p=probabilities)]
#             print(self._e)
        else:
            self._e = e0
        self._e_len = self._roadmap.graph[self._e[0]][self._e[1]]
        # current position on edge
        if x0 is None:
            self._x = random.random()
        else:
            self._x = x0
        self._v = v0
        self._sigma = sigma
        self._dt = dt
        self._name = name

    def predict(self):
        """Propogate the particle's state based on its dynamics and the roadmap

        When a particle is updated, it moves along the road segment by v0*dt, normalized by the
        length of its current road. If it reaches the end of the road (x >= 1), it queries the
        roadmap for other roads that connect to the current intersection and chooses one at
        random.
        """
        n = 0#np.random.normal(scale=self._sigma)
        self._x += (self._v + n)*self._dt/self._e_len

        if self._x >= 1.:
            dest_list = list(self._roadmap.graph[self._e[1]].keys())
            # no U-turns
            dest_list.remove(self._e[0])
            self._e = (self._e[1], random.choice(dest_list))
            self._e_len = self._roadmap.graph[self._e[0]][self._e[1]]
            self._x = 0.

        return self.state

    @property
    def state(self):
        return (self._e, self._x)

    @state.setter
    def state(self, new_state):
        e = new_state[0]
        x = new_state[1]
        try:
            self._roadmap.graph[e[0]][e[1]]
            self._e = e
        except KeyError:
            raise ValueError("Invalid roadmap edge.")
        if x < 0.:
            self._x = 0.
        elif x > 1.:
            self._x = 1.
        else:
            self._x = x

    @property
    def loc(self):
        return self._roadmap.get_loc(self.state)

    def __repr__(self):
        return '{}'.format(self._name)
        #         return '({:.2f} {:.2f}) ({:.2f} {:.2f}) {}'.format(self._e[0][0], self._e[0][1], self._e[1][0], self._e[1][1], self._x)
def DjikstraGraph(graph, initial_node):
    nodes = sorted(graph.keys())
    unvisited = sorted(graph.keys())
    visited = []
    dists = []
    paths = []
    for node in nodes:
        dists.append(np.inf)
        paths.append([])
        if node == initial_node:
            dists[-1] = 0
    while len(visited) < len(nodes):
        best_node = None
        best_node_val = -1
        for node in unvisited:
            if dists[nodes.index(node)] < best_node_val or best_node is None:
                best_node_val = dists[nodes.index(node)]
                best_node = node
        start_node = best_node
        visited.append(start_node)
        unvisited.remove(start_node)
        index_start = nodes.index(start_node)
        for node in graph[start_node].keys():
            index = nodes.index(node)
            new_dist = dists[index_start] + graph[start_node][node]
            if new_dist < dists[index]:
                dists[index] = new_dist
                paths[index] = paths[index_start]
                paths[index].append(node)
    return dists

def getShortestPath(graph, start_edge, start_percent, target_edge, target_percent, depth):
    nodes = sorted(graph.keys())
    start_index0 = nodes.index(start_edge[0])
    start_index1 = nodes.index(start_edge[1])
    end_index0 = nodes.index(target_edge[0])
    end_index1 = nodes.index(target_edge[1])
#     print(edge, target)
#     print(start_index0, start_index1, end_index0, end_index1)

    dist_from_start0 = DjikstraGraph(graph, start_edge[0])
    dist_from_start1 = DjikstraGraph(graph, start_edge[1])

    dist_to_start_node = [
        graph[start_edge[0]][start_edge[1]]*start_percent,
        graph[start_edge[1]][start_edge[0]]*(1-start_percent)]

    dist_to_point = [
        graph[target_edge[0]][target_edge[1]]*target_percent,
        graph[target_edge[1]][target_edge[0]]*(1-target_percent)]

    distances = [
        dist_to_start_node[0] + dist_from_start0[end_index0] + dist_to_point[0],
        dist_to_start_node[0] + dist_from_start0[end_index1] + dist_to_point[1],
        dist_to_start_node[1] + dist_from_start1[end_index0] + dist_to_point[0],
        dist_to_start_node[1] + dist_from_start1[end_index1] + dist_to_point[1]
    ]
    if start_edge == target_edge:
        distances.append(graph[start_edge[0]][start_edge[1]]*abs(start_percent-target_percent))
    elif (start_edge[1], start_edge[0]) == target_edge:
        distances.append(graph[start_edge[0]][start_edge[1]]*abs((1-start_percent)-target_percent))

    return min(distances)

def getAvgDistance(r):
    vals = []
    for node in sorted(r.graph.keys()):
        avg_dist = np.mean(DjikstraGraph(r.graph, node))
        vals.append(avg_dist)

    return np.mean(vals)
##Map Types
def createGridLayout(x,y,min_edge_length, max_edge_length):
    nodes = []
    edges = []
    for i in range(y):
        for j in range(x):
            x_val = 0
            y_val = 0
            if i > 0:
                y_val = nodes[(i-1)*x + j][1]
            if j > 0:
                x_val = nodes[i*x+j-1][0]
#                 print(nodes[i*x+j-1][0],nodes[(i-1)*x + j][1])
            nodes.append((
                x_val + np.random.uniform(low=min_edge_length, high=max_edge_length),
                y_val + np.random.uniform(low=min_edge_length, high=max_edge_length)))
#     print(nodes)
    for i in range(y):
        for j in range(x-1):
            edges.append((j+x*i,j+1+x*i))
#             print(j,i, y,y*i, (j+(x)*i,j+1+(x)*i))

    for i in range(y-1):
        for j in range(x):
            edges.append((j+x*i,j+x*(i+1)))
#             print((j+(x)*(i),j+(x)*(i+1)))
#     print(edges)
    return [nodes, edges]

def getSequences(graph, t0, targets, max_depth):
    #Base Case: 1 target in targets
    if len(targets) == 1:
        t1 = targets[0]
        dist = getShortestPath(graph, t0._e, t0._x, t1._e, t1._x, max_depth)
        return [[dist, [targets[0]], [dist]]]
    sequence_info = []
    for t1 in targets:
        unvisited = [target for target in targets if target != t1]
        dist = getShortestPath(graph, t0._e, t0._x, t1._e, t1._x, max_depth)
        sequence_data = getSequences(graph, t1, unvisited, max_depth)
        for entry in sequence_data:
            entry[0] += dist
            entry[1].insert(0, t1)
#             entry[2].insert(0, (t0, t1, dist))
            entry[2].insert(0, dist)
#             print(t0, t1, dist)
#             print(dist)
#             print(entry)
        sequence_info.extend(sequence_data)
    return sequence_info

def getShortestRoundTrip(graph, targets, max_depth):
    # print(len(targets))
#     target1 = targets[0]
#     target2 = targets[1]
#     target3 = targets[2]
#     target4 = targets[3]
    min_dist = np.inf
    for t1 in targets:
        unvisited = [target for target in targets]
        unvisited.remove(t1)
        sequences = getSequences(graph, t1, unvisited, max_depth)
        for sequence in sequences:
            return_dist = getShortestPath(graph, t1._e, t1._x, sequence[1][-1]._e, sequence[1][-1]._x, max_depth)
            sequence[0] += return_dist
            sequence[1].insert(0,t1)
            sequence[2].append(return_dist)
            min_dist = min(min_dist, sequence[0])
            # print('sequence',sequence)

#         print('lists', t1._e, t1._x, [(t._e, t._x) for t in unvisited])

#     dist1_2 = getShortestPath(graph, target1._e, target1._x, target2._e, target2._x, max_depth)
#     dist2_3 = getShortestPath(graph, target2._e, target2._x, target3._e, target3._x, max_depth)
#     dist3_4 = getShortestPath(graph, target3._e, target3._x, target4._e, target4._x, max_depth)
#     dist4_1 = getShortestPath(graph, target4._e, target4._x, target1._e, target1._x, max_depth)
#     route1 = dist1_2 + dist2_3 + dist3_4 + dist4_1

#     dist1_3 = getShortestPath(graph, target1._e, target1._x, target3._e, target3._x, max_depth)
#     dist3_4 = getShortestPath(graph, target3._e, target3._x, target4._e, target4._x, max_depth)
#     dist4_2 = getShortestPath(graph, target4._e, target4._x, target2._e, target2._x, max_depth)
#     dist2_1 = getShortestPath(graph, target2._e, target2._x, target1._e, target1._x, max_depth)
#     route2 = dist1_3 + dist3_4 + dist4_2 + dist2_1

#     dist1_4 = getShortestPath(graph, target1._e, target1._x, target4._e, target4._x, max_depth)
#     dist4_2 = getShortestPath(graph, target4._e, target4._x, target2._e, target2._x, max_depth)
#     dist2_3 = getShortestPath(graph, target2._e, target2._x, target3._e, target3._x, max_depth)
#     dist3_1 = getShortestPath(graph, target3._e, target3._x, target1._e, target1._x, max_depth)
#     route3 = dist1_4 + dist4_2 + dist2_3 + dist3_1

#     print('0-1: {:.2f}\t1-2: {:.2f}\t2-3: {:.2f}\t3-0: {:.2f}\n0-2: {:.2f}\t2-3: {:.2f}\t3-1: {:.2f}\t1-0: {:.2f}\n0-3: {:.2f}\t3-1: {:.2f}1-2: {:.2f}\t2-0: {:.2f} '.format(
#         dist1_2, dist2_3, dist3_4, dist4_1,
#         dist1_3, dist3_4, dist4_2, dist2_1,
#         dist1_4, dist4_2, dist2_3, dist3_1
#     ))

    return min_dist

def getAverageDistanceOverSim(v0, dt, r, duration=40, num_targets=2, max_depth=7, est_dist=None):
#     target1 = Particle(r, v0=v0, dt=dt, sigma=4)
#     target2 = Particle(r, v0=v0, dt=dt, sigma=4)
    targets = []
    for i in range(num_targets):
        targets.append(Particle(r, v0=v0, dt=dt, sigma=4, name=i))
    total_distance = np.zeros(num_targets - 2)
    shortest_path_values = np.zeros(num_targets-2)
#     print(targets)
    with trange(int(duration/dt), leave=False) as t:
        for j in t:

    # for i in tqdm(range(duration), desc='sim', leave=False):
#         print('test', i)
#         shortest_path_value = getShortestPath(r.graph, target1._e, target1._x, r, target2._e, target2._x, max_depth)
            for idx in range(2, num_targets):
                # print('num targets',i)
                shortest_path_value = getShortestRoundTrip(r.graph, targets[:idx], max_depth)
                # print('shortest path',shortest_path_value, shortest_path_value/idx)
                shortest_path_values[idx-2] = shortest_path_value/idx
                # print(idx, idx-2, shortest_path_value, shortest_path_values)
                # total_distance[i-2] += shortest_path_value/i
            total_distance += shortest_path_values
            # print(total_distance)
            # print('total distance', total_distance)
    #         print('dist',total_distance)
    #         target1.predict()
    #         target2.predict()
            for target in targets:
                target.predict()
            desc = ["{:.0f}".format(dist/(1.0+j)) for dist in total_distance]
            t.set_description("Simulation {} {}".format(est_dist, desc))
    # print('total distance',total_distance/duration)
    return total_distance/(duration/dt)

def getAverageErrorForLayoutParallel(layout, edge_length, v0, dt, num_runs, sim_duration, num_targets):
    nodes, edges = createGridLayout(layout[0], layout[1], edge_length, edge_length)
    r = Roadmap(nodes, edges)
    with Pool(processes=num_runs) as pool:
        data = pool.starmap(getAverageDistanceOverSim, [[v0, dt, r,
            sim_duration, num_targets, 7] for idx in range(num_runs)])
        avg_distance = np.array(data)
        # np.save("data_{}x{}-{}-{}_h".format(x,y,int(dist_e), entropy_resolution), data_h)
    # print(avg_distance)
    print(avg_distance.shape)
    avg_distance = np.mean(avg_distance, axis=0)

    estimated_dist = getAvgDistance(r)
#     print(estimated_dist, total_distance/num_runs, num_runs)
    error = estimated_dist*np.ones(num_targets - 2) - avg_distance#total_distance/num_runs
    print(estimated_dist, avg_distance, error)
    return error

def getAverageErrorForLayout(layout, edge_length, v0, dt, num_runs, sim_duration, num_targets):
    nodes, edges = createGridLayout(layout[0], layout[1], 0, edge_length)
    r = Roadmap(nodes, edges)
    estimated_dist = ['{:.0f}'.format(getRoundTripEst(idx,getAvgDistance(r))) for idx in range(2,num_targets)]
    # print(nodes)
    # print(edges)
    # print(estimated_dist)
    distances = None
    total_distance = np.zeros(num_targets - 2)
    with trange(num_runs, leave=False) as t:
        for j in t:
    # for j in tqdm(range(num_runs), desc='runs', leave=False):
            avg_distance = getAverageDistanceOverSim(v0, dt, r, duration=sim_duration, num_targets=num_targets, est_dist=estimated_dist)
            if distances is None:
                distances = avg_distance.reshape((1,len(avg_distance)))
            else:
                distances = np.vstack((distances, avg_distance))
            # total_distance += avg_distance
            avg_distances = np.mean(distances,axis=0)
            # print(estimated_dist, total_distance/(1+j))
            # print(distances)
            # print(avg_distances)
            sim_dist_str = ["{:.0f}".format(dist) for dist in avg_distances]
            t.set_description("Estimated Dist: {} Sim Dist: {}".format(
                estimated_dist, sim_dist_str) )
            np.save("averages", distances)

    error = estimated_dist*np.ones(num_targets - 2) - total_distance/num_runs
    return error

def getAverageErrorForLayoutType(layout, edge_length, v0, dt, num_iterations, num_runs, sim_duration, num_targets):
    dt = .1
    total_error = np.zeros(num_targets - 2)
    for i in tqdm(range(num_iterations)):
        error = getAverageErrorForLayout(layout, edge_length, v0, dt, num_runs, sim_duration, num_targets=num_targets)
#         print(error)
        total_error += error
    return total_error/num_iterations
def main():
    # for i in range(1,5):
    #     edge_length = 100*i
    #     for layout in layouts:
    #         dt = .1
    #         getAverageErrorForLayoutType((2,2), 100, 0, .1, 100, 1000, 1)

    ## Get average distance between any two particles on given map
    Va = 30
    data = []
    for i in reversed(range(1,2)):
        dist_e = i*100
        print('\n\ndist', dist_e)
    #     layout_descriptions = [
    #         "2x2", '2x3', '2x4', '2x5', '2x6', '2x7', '2x8', '2x9',
    #         '3x3', '3x4', '3x5', '3x6', '3x7', '3x8', '3x9',
    #         '4x4', '4x5', '4x6', '4x7', '4x8', '4x9',
    #         '5x5','5x6', '5x7', '5x8', '5x9',
    #         '6x6', '6x7', '6x8', '6x9',
    #         '7x7', '7x8', '7x9',
    #         '8x8', '8x9',
    #         '9x9'
    #     ]
        layouts = [
            # (2,2)#,(2,3),(2,4),(2,5),(2,6),(2,7),(2,8),(2,9),
            (3,3),#(3,4),(3,5),(3,6),(3,7),(3,8),(3,9),
    #          (4,4),(4,5),(4,6),(4,7),(4,8),(4,9),
            # (5,5),#(5,6),(5,7),(5,8),(5,9),
    #         (6,6),#(6,7),(6,8),(6,9),
    #         (7,7),#(7,8),(7,9),
    #         (8,8),#(8,9),
    #         (9,9)
        ]

        for layout in layouts:
            print(layout)
            dt = .5

    #         total_value = 0
            num_iterations=100
            num_runs = 100
            num_targets = 9
            v0 = 10
            sim_duration = 300
            total_error = np.zeros(num_targets-2)#[0 for i in range(num_targets)]
    #         sim_time = 0
    #         max_depth = 10
            with trange(num_iterations, leave=False) as t:
                for i in t:
            # for i in tqdm(range(num_iterations), desc='iterations'):
                    error = getAverageErrorForLayout(layout, dist_e, v0, dt, num_runs,
                                                     sim_duration, num_targets)
                    total_error += error
                    # print('averaged error',total_error/(i+1))
                    desc = ["{:.1f}".format(err/(i+1)) for err in total_error]
                    t.set_description("Iterations {}".format(desc))
            data.append([
                dist_e,
                layout,
    #             total_value/num_runs,
    #             getAvgDistance(r),
    #             abs(total_value/num_runs - getAvgDistance(r)),
                np.array(total_error) /num_iterations
    #             dist_e*len(edges)
            ])
            print("EdgeLength: {} Layout: {} error: {}".format(
                data[-1][0], data[-1][1], data[-1][2]
            ))

def getTriangle(side0, side1, angle2):
    side2 = np.sqrt(side0**2+side1**2-2*side0*side1*np.cos(angle2))
    # print(np.sin(angle2)*side0/side2, side0, side2)
    angle0 =  np.arccos((side1**2+side2**2-side0**2)/(2*side1*side2))
    return side2, angle0

def getRoundTripEst(n, side):
    if n == 2:
        return n*side
    if n < 2:
        return -1
    sides_from_vertex = [side]
    num_diagonals = int(2*n*(n-3)/2/n)
    side0 = side
    side1 = side
    corner_angle = (n-2)*np.pi/n
    # print(corner_angle*180/np.pi)
    angle2 = corner_angle
    for idx in range(num_diagonals):
        side2, angle0 = getTriangle(side0, side1, angle2)
        angle1 = np.pi-angle2-angle0
        sides_from_vertex.append(side*np.sin(angle1)/np.sin(angle2))
        side0 = side2
        side1 = side
        angle2 = corner_angle - angle0
    sides_from_vertex.append(side)
    # print(sides_from_vertex)
    # print('avg', np.mean(sides_from_vertex))
    return n*np.mean(sides_from_vertex)


if __name__ == '__main__':
    main()
    # print(['{:.2f}'.format(getDiagonals(idx,100)) for idx in range(2,8)])
    # print(157/2, 246/3, 286/4, 324/5,345/6, 354/7)
    # print(getRoundTripEst(4,100))

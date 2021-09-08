import numpy as np
from math import atan2
# import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import copy
import random
# from simulator import calc_entropy

class AgentJupyter(object):
    def __init__(self, speed, roadmap, dt, fov=30., e0=None, x0=None, entropy_resolution=None):

        if e0 is None:
            options = [(np.array(start), np.array(dest)) for start in roadmap.graph.keys()
                          for dest in roadmap.graph[start].keys()]
            e0 = random.choice(options)
        x0 = random.random() if x0 is None else x0
        self.pos = e0[0] + x0*(e0[1] - e0[0])

        self._roadmap = roadmap
        # dest = self._roadmap.get_nearest_waypoint(self.pos)
        print(e0, self.pos)
        self.psi = atan2(e0[1][0] - self.pos[0], e0[1][1] - self.pos[1])
        self._dt = dt
        self._t = 0
        self._sc_agent = None
        self._sc_fov = None
        self.speed = speed
        self.fov = fov
        self.path = [(e0[0][0],e0[0][1]), (e0[1][0],e0[1][1])]
        self.current_dest = 1
        self.discount = .8
        self.edge_distance = 100
        self._entropy_resolution = entropy_resolution
    #
    def update_pos(self):
        self._t += self._dt
        distance = np.array((
            self.path[self.current_dest][0] - self.pos[0],
            self.path[self.current_dest][1] - self.pos[1]))


        if abs(distance[0]) > abs(distance[1]):
            distance[1] *= self.edge_distance*.2
        elif abs(distance[1]) > abs(distance[0]):
            distance[0]*= self.edge_distance*.2

        vel = self.speed * (distance / np.linalg.norm(distance) )
        self.pos = self.pos + np.array(vel * self._dt)

        if self._sc_agent is not None:
            self.update_plot()
        return distance
    #

    def update(self, pfs):
        distance = self.update_pos()
        if np.linalg.norm(distance) < (self.speed*self._dt+2):
            if self.current_dest == len(self.path)-1:
                self.update_path(pfs)
            self.current_dest += 1
            self.update_edge_distance_()
        #
    #
    def update_edge_distance_(self):
        self.edge_distance = np.linalg.norm(
            np.array(self.path[self.current_dest-1]) - np.array(self.path[self.current_dest]))
        #
    #
    def update_path(self):
        pass
    #
    def init_plot(self, ax):
        self._sc_agent = ax.scatter([self.pos[0]], [self.pos[1]], s=50, marker='D', facecolor='red', label='agent')
        self._sc_fov = plt.Circle((self.pos[0], self.pos[1]), self.fov, facecolor='None', edgecolor='orange')
        ax.add_patch(self._sc_fov)
    #
    def update_plot(self):
        self._sc_agent.set_offsets([self.pos])
        self._sc_fov.center = self.pos[0], self.pos[1]
    #
class AgentJupyterRandom(AgentJupyter):
    def __init__(self, speed, roadmap, dt, fov=30., e0=None, x0=None, entropy_resolution=1):
        super(AgentJupyterPerfect, self).__init__(speed, roadmap, dt, fov=fov, e0=e0, x0=x0, entropy_resolution=entropy_resolution)
    #
    def update_path(self, pfs):
        destinations = list(self._roadmap.graph[self.path[self.current_dest]].keys())
        self.path.append(random.choice(destinations))
    #
#
class AgentJupyterGreedy(AgentJupyter):
    def __init__(self, speed, roadmap, dt, fov=30., e0=None, x0=None, entropy_resolution=1):
        super(AgentJupyterPerfect, self).__init__(speed, roadmap, dt, fov=fov, e0=e0, x0=x0, entropy_resolution=entropy_resolution)

    def update_path(self, pfs):
        target_weights = np.array([1 for pf in pfs])

        options = list(self._roadmap.graph[(self.end_node[0], self.end_node[1])].keys())
        best_value = -1
        best_option = None
        for option in options:
            value = self.get_edge_value(self.end_node, option, target_weights, pfs)
            value += self.get_greedy_path_value(option, target_weights, pfs, lookahead)
            if value > best_value:
                best_value = value
                best_option = option
        self.path.append(best_option)

    def get_edge_value(self, start, dest, target_weights, pfs):
        edge_value = 0
        for ii in range(len(target_weights)):
            value = len(np.where((
                    pfs[ii,:,2:6]==(start[0], start[1], dest[0], dest[1])
                ).all(axis=1))[0]) + len(np.where((
                    pfs[ii,:,2:6]==(dest[0], dest[1], start[0], start[1])
                ).all(axis=1))[0])
            edge_value += value*target_weights[ii]
        return edge_value
    #
    def get_greedy_path_value(self, start, target_weights, pfs, lookahead, weighting=.7):
        if lookahead == 0:
            return 0
        destinations = list(self._roadmap.graph[start].keys())
        best_option = None
        best_value = -1
        for dest in destinations:
            edge_value = self.get_edge_value(start, dest, target_weights, pfs)
            if edge_value > best_value:
                best_value = edge_value
                best_option = dest
        return best_value + weighting*self.get_greedy_path_value(best_option, target_weights, pfs, lookahead - 1)
    #
#
class AgentJupyterExhaustive(AgentJupyter):
    def __init__(self, speed, roadmap, dt, fov=30., e0=None, x0=None, entropy_resolution=1):
        super(AgentJupyterPerfect, self).__init__(speed, roadmap, dt, fov=fov, e0=e0, x0=x0, entropy_resolution=entropy_resolution)

    def update_path(self, pfs):
        X = np.array([pfs[idx].X for idx in range(len(pfs))])
        entropy = self.calc_entropy(X, res=self._entropy_resolution)
        edges = {}
        edge_vals = {}

        norm = np.sum(entropies)
        target_weights = np.array([e/norm for e in entropies])
        target_weights = 1 / (1 + np.exp(-10*(target_weights - 0.5)))

        N = [pf.X.shape[0] for pf in pfs]

        for ii in range(len(pfs)):
            avg_den, min_den = pfs[ii].get_max_particle_density()
            for jj in range(len(pfs[ii].X)):
                edge = ((pfs[ii].X[jj][2], pfs[ii].X[jj][3]),
                        (pfs[ii].X[jj][4], pfs[ii].X[jj][5]))
                #
                if edge not in edges:
                    edges[edge] = []
                    edge_vals[edge] = [0,0,0]
                #
                particle = copy.deepcopy(pfs[ii].X[jj,:])
                particle = np.append(particle, ii)
                edges[edge].append(particle)
                edge_vals[edge][0] += 1
                edge_vals[edge][ii+1] += 1
            #
        #
        particle_map, best_path, best_value = self.build_particle_map(
            (self.end_node[0], self.end_node[1]),
            edges, target_weights, N, 0, 0, 6)
        #
        # print(best_path)
        self.path.append(best_path[0])

    def calc_entropy(self, X, res=1):
        """Returns the entropy of the estimate in nats

            r -- roadmap graph on which the particles exist
            X -- state of each particle, shape=(M, N, 12),
                 M = number of targets
                 N = number of particles
            """
        M = X.shape[0]
        N = X.shape[1]
        dists = np.linalg.norm(X[:,:, :2] - X[:,:, 3:5], axis=-1)
        h = np.zeros(M)
        nodes_visited = []
        for start in self._roadmap.graph.keys():
                for end in self._roadmap.graph[start].keys():
                    length = self._roadmap.graph[start][end]
                    on_edge = np.all(X[:,:, 3:7] == start + end, axis=-1)
                    for idx in range(M):
                        bin_start = 0.0
                        while bin_start < length:
                            in_bin = np.all([dists >= bin_start, dists <= bin_start + res], axis=0)
                            count = np.sum(np.all([on_edge[idx], in_bin[idx]], axis=0))
                            p = count / (N)
                            if p > 0:
                                h[idx] -= p*np.log(p)
                            bin_start += res
        return h
    #

class AgentJupyterPerfect(AgentJupyter):
    def __init__(self, speed, roadmap, dt, fov=30., e0=None, x0=None, entropy_resolution=1):
        print(e0,x0)
        super(AgentJupyterPerfect, self).__init__(speed, roadmap, dt, fov=fov, e0=e0, x0=x0, entropy_resolution=entropy_resolution)
        self.target_index = 0

    def getBestPath(self,target):
        # option_a = self.path[self.current_dest]
        # option_b = self.path[self.current_dest-1]
        # option_a_dist = np.array((
        #     self.path[self.current_dest][0] - self.pos[0],
        #     self.path[self.current_dest][1] - self.pos[1]))
        # option_b_dist = np.array((
        #     self.path[self.current_dest-1][0] - self.pos[0],
        #     self.path[self.current_dest-1][1] - self.pos[1]))
        # target_a = target._e[0]
        # target_b = target._e[1]
        # target_a_dist = (target._e[1]-target._e[0])*target._x
        # target_b_dist = (target._e[1]-target._e[0])*(1.0-target._x)

        # if the current destination is on the same edge as the target keep moving
        # if the last destination is on the edge with the target turn around
        current_dest = self.path[self.current_dest]
        last_dest = self.path[self.current_dest-1]
        distance = np.array((
            current_dest[0] - self.pos[0],
            current_dest[1] - self.pos[1]))
        distance_reverse = np.array((
            last_dest[0] - self.pos[0],
            last_dest[1] - self.pos[1]))

        # print(last_dest, current_dest, target._e)

        if last_dest in target._e and current_dest not in target._e:
            # if the target is on a road touching the previous intersection but not on the same intersection
            return last_dest
        elif last_dest in target._e and current_dest in target._e:
            #if the target is on the same road segment as the agent
            if (last_dest, current_dest) == target._e:
                target_percent = target._x
                # print('same direction', target_percent)

            else:
                target_percent = 1-target._x
                # print('opposite', target_percent)


            agent_percent = 1-np.linalg.norm(distance)/self.edge_distance
            # print('agent percent', agent_percent)
            if agent_percent > target_percent:
                return last_dest
            else:
                return current_dest

        elif last_dest not in target._e and current_dest in target._e:
            return current_dest

        # if the target isn't on the same edge or adjacent edge then check if which connecting node is the closest


        paths, dists = self.djikstraGraph(current_dest)
        target_a = target._e[0]
        target_a_dist = self._roadmap.graph[target._e[0]][target._e[1]] * target._x
        target_b = target._e[1]
        target_b_dist = self._roadmap.graph[target._e[0]][target._e[1]] * (1.0-target._x)

        c2a_path = paths[sorted(self._roadmap.graph.keys()).index(target_a)][0]
        c2a_dist = dists[sorted(self._roadmap.graph.keys()).index(target_a)] + np.linalg.norm(distance) +target_a_dist
        c2b_path = paths[sorted(self._roadmap.graph.keys()).index(target_b)][0]
        c2b_dist = dists[sorted(self._roadmap.graph.keys()).index(target_b)] + np.linalg.norm(distance_reverse) + target_b_dist
        c_min = min(c2a_dist,c2b_dist)

        paths, dists = self.djikstraGraph(last_dest)
        l2a_path = paths[sorted(self._roadmap.graph.keys()).index(target_a)][0]
        l2a_dist = dists[sorted(self._roadmap.graph.keys()).index(target_a)] + np.linalg.norm(distance_reverse) + target_b_dist
        l2b_path = paths[sorted(self._roadmap.graph.keys()).index(target_b)][0]
        l2b_dist = dists[sorted(self._roadmap.graph.keys()).index(target_b)] + np.linalg.norm(distance_reverse) + target_b_dist
        l_min = min(l2a_dist,l2b_dist)
        # print(l_min, c_min)
        if l_min < c_min:
            return last_dest
        else:
            return current_dest

        #
        # paths, dists = self.djikstraGraph(option_b)

    def getTargetIndex(self,targets):
        target = targets[self.target_index]
        target_pos = np.array(target._e[0]) + (np.array(target._e[1])-np.array(target._e[0]))*target._x
        distance = np.linalg.norm(target_pos - np.array(self.pos))
        if distance < (self.speed*self._dt+2):
            self.target_index = not self.target_index
        return self.target_index


    def update(self, pfs, targets):
        '''
        Verify that we are at an intersection.
        Find the shortest path to the closest node to the target
        go down the first road on the path to the closest node.
        Repeat
        '''
        X = np.array([pfs[idx].X for idx in range(len(pfs))])
        entropy = self.calc_entropy(X, res=self._entropy_resolution)
        options = np.where(entropy==max(entropy))[0]

        target_index = self.getTargetIndex(targets)
        target = targets[target_index]
        distance = self.update_pos()

        np.set_printoptions(precision=3)
        target_edge = target._e

        next_dest = self.getBestPath(target)

        if next_dest != self.path[self.current_dest]:
            self.path.append(next_dest)
            self.current_dest += 1
            self.update_edge_distance_()
            distance = np.array((
                self.path[self.current_dest][0] - self.pos[0],
                self.path[self.current_dest][1] - self.pos[1]))

        if np.linalg.norm(distance) < (self.speed*self._dt+10):
            if self.current_dest == len(self.path)-1:
                nearest_node = target_edge[int(target._x>.5)]

                current_dest = self.path[self.current_dest]
                paths, dists = self.djikstraGraph(current_dest)
                if current_dest in target_edge:
                    self.path.append(target_edge[1-target_edge.index(current_dest)])
                else:
                    self.path.append(paths[sorted(self._roadmap.graph.keys()).index(nearest_node)][0])
                self.current_dest += 1
                self.update_edge_distance_()

    def djikstraGraph(self, initial_node):
        graph = self._roadmap.graph
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
                    paths[index].extend(paths[nodes.index(best_node)])
                    paths[index].append(node)
        return paths, dists
    def calc_entropy(self, X, res=1):
        """Returns the entropy of the estimate in nats

            r -- roadmap graph on which the particles exist
            X -- state of each particle, shape=(M, N, 12),
                 M = number of targets
                 N = number of particles
            """

        ## x, y, speed, start_x, start_y, end_x, end_y, direction_x, direction_y, distance, sigma, w
        # print(res)
        M = X.shape[0]
        N = X.shape[1]
        # calculate the distance of each particle from the beginning of its road segment
        dists = np.linalg.norm(X[:,:, :2] - X[:,:, 3:5], axis=-1)

        h = np.zeros(M)
        nodes_visited = []
        for start in self._roadmap.graph.keys():
                for end in self._roadmap.graph[start].keys():
                    length = self._roadmap.graph[start][end]
                    # find the particles on this road segment
                    on_edge = np.all(X[:,:, 3:7] == start + end, axis=-1)
                    for idx in range(M):
                        bin_start = 0.0
                        while bin_start < length:
                            in_bin = np.all([dists >= bin_start, dists <= bin_start + res], axis=0)

                            count = np.sum(np.all([on_edge[idx], in_bin[idx]], axis=0))# + np.sum(np.all([on_edge, in_bin_reverse], axis=0))
                            p = count / (N)
                            if p > 0:
                                h[idx] -= p*np.log(p)
                            bin_start += res
        return h#, get_pos_var(X, hist), counts, probs

class AgentROS(object):
    def __init__(self, center, width, height, speed, period, roadmap, dt, fov=30):
        self._c = center
        self._w = width
        self._h = height
        self._period = period
        self.pos = np.array(center)
        self._roadmap = roadmap
        dest = self._roadmap.get_nearest_waypoint(self.pos)
        self.psi = atan2(dest[0] - self.pos[0], dest[1] - self.pos[1])
        self._dt = dt
        self._t = 0
        self._sc_agent = None
        self._sc_fov = None
        self.speed = speed
        self.fov = fov
        self.path = [dest]
        self.current_dest = 0
        self.discount = .5
        self.start_node = self.pos
        self.end_node = np.array(dest)
        self.segment_distance = np.linalg.norm(self.end_node - self.start_node)
        self.dist_to_intersection = 20
    #
    def update(self, pfs):
        self._t += self._dt
        distance = self.end_node - self.pos

        vel = self.speed * (distance / np.linalg.norm(distance) )
        self.pos = self.pos + np.array(vel * self._dt)
        if self._sc_agent:
            self.update_plot()
    #
    def is_at_intersection(self):
        distance_travelled = np.linalg.norm(self.pos - self.start_node)
        return distance_travelled >= self.segment_distance
    #
    def get_options(self):
        options = sorted(self._roadmap.graph.keys())
        options.remove((self.end_node[0], self.end_node[1]))
        return options

    def set_next_intersection(self, action):
        self.start_node = self.end_node
        self.end_node = np.array(action)

        self.segment_distance = np.linalg.norm(self.end_node - self.start_node)
    #
    def get_next_waypoint(self, entropies, pfs):
        edges = {}
        edge_vals = {}

        norm = np.sum(entropies)
        # print(norm)
        target_weights = np.array([e/norm for e in entropies])
        target_weights = 1 / (1 + np.exp(-10*(target_weights - 0.5)))

        N = [pf.X.shape[0] for pf in pfs]

        for ii in range(len(pfs)):
            avg_den, min_den = pfs[ii].get_max_particle_density()
            for jj in range(len(pfs[ii].X)):
                edge = ((pfs[ii].X[jj][2], pfs[ii].X[jj][3]),
                        (pfs[ii].X[jj][4], pfs[ii].X[jj][5]))
                #
                if edge not in edges:
                    edges[edge] = []
                    edge_vals[edge] = [0,0,0]
                #
                particle = copy.deepcopy(pfs[ii].X[jj,:])
                particle = np.append(particle, ii)
                edges[edge].append(particle)
                edge_vals[edge][0] += 1
                edge_vals[edge][ii+1] += 1
            #
        #
        particle_map, best_path, best_value = self.build_particle_map(
            (self.end_node[0], self.end_node[1]),
            edges, target_weights, N, 0, 0, 6)
        #
        # print(best_path)
        return best_path[0]
    #
    def build_particle_map(self, start, edges, target_weights, N, start_value, lookahead, lookahead_max):
        if lookahead == lookahead_max:
            return start_value, [], start_value
        #
        graph = {}

        best_path = []
        best_value = -1

        for destination in self._roadmap.graph[start]:
            edge0 = (start, destination)
            edge1 = (destination, start)
            value = 0
            if edge0 in edges:
                for ii in range(len(target_weights)):
                    idx = np.isin(np.vstack(edges[edge0])[:,10], np.array(ii))
                    val = np.sum(idx)/N[ii]
                    value += target_weights[ii] * val * (self.discount**lookahead)
                #
            #
            if edge1 in edges:
                for ii in range(len(target_weights)):
                    idx = np.isin(np.vstack(edges[edge1])[:,10], np.array(ii))
                    val = np.sum(idx)/N[ii]
                    value += target_weights[ii] * val * (self.discount**lookahead)
                #
            #

            new_edges = {}
            for edge in edges:
                if edge == edge0 or edge == edge1:
                    value += len(edges[edge][0]) * (self.discount**lookahead)
                    pass
                else:
                    particles = np.array(copy.deepcopy(edges[edge]))
                    particles[:,0:2] = particles[:,0:2] + particles[:,6:8] * self._dt
                    update_vector = np.linalg.norm(particles[:,0:2] - particles[:,2:4]) > particles[:,8]
                    for particle in particles[update_vector]:
                        a = tuple(particle[2:4])
                        b = tuple(particle[4:6])

                        dest_list = list(self._roadmap.graph[b].keys())
                        dest_list.remove(a)
                        a = particle[4:6]
                        b = np.array(random.choice(dest_list))
                        vector = b - a
                        distance = np.linalg.norm(vector)
                        vector = np.linalg.norm(particle[6:8]) * vector/distance
                        particle[0:9] = [a[0], a[1], a[0], a[1], b[0],
                                       b[1], vector[0], vector[1], distance]
                        #
                        new_edge = ((particle[2], particle[3]), (particle[4], particle[5]))
                        if new_edge not in new_edges:
                            new_edges[new_edge] = []
                        #
                        new_edges[new_edge].append(particle)
                    #
                #
            #
            graph[edge0], path, path_value = self.build_particle_map(
                destination, new_edges, target_weights, N, start_value + value, lookahead+1, lookahead_max)
            #
            # print(path_value)
            if path_value > best_value:
                path.insert(0, destination)
                best_path = path
                best_value = path_value
            #
        #
        return graph, best_path, best_value
    #
    def init_plot(self, ax):
        self._sc_agent = ax.scatter([self.pos[0]], [self.pos[1]], s=50,
                                    marker='D', facecolor='red', label='agent')
        #
        self._sc_fov = plt.Circle((self.pos[0], self.pos[1]), self.fov,
                                  facecolor='None', edgecolor='orange')
        #
        ax.add_patch(self._sc_fov)
    #
    def update_plot(self):
        self._sc_agent.set_offsets([self.pos])
        self._sc_fov.center = self.pos[0], self.pos[1]
    #
#

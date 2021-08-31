import sys
sys.path.insert(0, "/home/mouse/projects/magicc/mass/algorithms")

##### from __future__ import division
from multiprocessing import Pool
import time
import random
import numpy as np
import scipy.io as sio
#import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
from math import atan2, log
#from tqdm import trange, tqdm
import copy
import datetime
#from collections import Counter

# from importlib import reload
# import roadmap
# reload(roadmap)
# from particle import Particle
# from roadmap import Roadmap
# import pf
# reload(pf)
# from pf import PFJupyter as PF
# import agent
# reload(agent)
# from agent import AgentJupyter as Agent
# from agent import AgentJupyterRandom as AgentRandom
# from agent import AgentJupyterPerfect as AgentPerfect

# Variance and Entropy calculations
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

class PF(object):
    def __init__(self, roadmap, num_particles, dt, v0=10., sigma=4, x0=None, P_fa=0.1, P_miss=0.05):
        self._roadmap = roadmap
        self._N = num_particles
        self._v0 = v0
        # particle shape
        ## x, y, speed, start_x, start_y, end_x, end_y, direction_x, direction_y, distance, sigma, w
        self.X = np.ndarray(shape=(self._N, 12))

        if x0 is None:
            #uniformly distribute the particles on the roadway
            for i in range(self._N):
                a = random.choice(list(self._roadmap.graph.keys()))
                b = np.array(random.choice(list(self._roadmap.graph[a].keys())))
                a = np.array(a)
                vector = b - a
                loc = a + vector * random.random()
                distance = np.linalg.norm(vector)
                vector = vector/distance
                self.X[i] = [loc[0], loc[1], v0, a[0], a[1], b[0],
                                      b[1], vector[0], vector[1], distance, sigma, 1/self._N]

        else:
            raise NotImplementedError

        self.best = self.X[0]
        self._dt = dt
        self._P_fa = P_fa
        self._P_miss = P_miss

    def get_particle_distance_from_target(self, loc, acceptable_distance):
        avg = 0
        num_in_distance = 0
        dist = np.linalg.norm(self.X[:,0:2] - loc, axis=1)
        num_in_distance = len(np.where(dist < acceptable_distance)[0])
        avg = np.average(dist)
        return avg, num_in_distance

    def get_max_particle_density(self):
        overall_avg = 0
        min_avg = 9999999999
        for i in range(len(self.X)):
            dist = np.linalg.norm(self.X[:,0:2] - self.X[i,0:2], axis=1)
            avg_dist = np.average(dist)
            overall_avg += avg_dist
            if avg_dist < min_avg:
                min_avg = avg_dist
        return overall_avg, min_avg

    def get_edge_particle_density(self):
        edges = {}
        edges_present, count = np.unique(self.X[:, 3:7], return_counts=True, axis=0)
        for i in range(len(edges_present)):
            start_edge = (edges_present[i][0], edges_present[i][1])
            end_edge = (edges_present[i][2], edges_present[i][3])
            edges[(start_edge, end_edge)] = count[i]
        return edges

    def get_measurement_likelihood(self, z, R):
        return np.sum(mvn.pdf(self.X[:,:2], z, R))

    def low_var_sample(self):
        M = self._N
        r = np.random.uniform(0,1/M)
        c = self.X[0,11]
        new_particles = np.zeros_like(self.X)
        i = 0
        last_i = 1
        unique = 1
        insert_index = 0
        for m in range(M):
            u = r + m/M
            while u > c:
                i += 1
                c = c + self.X[i,11]
            new_particles[insert_index] = copy.deepcopy(self.X[i])
            insert_index += 1
            if last_i != i:
                unique += 1
            last_i = i
        self.X = new_particles
        return unique

    def predict(self, timestep=1):
#         old_particles = deepcopy(self.X)
        n = np.random.normal(scale=self.X[:,10:11])
        loc = self.X[:,0:2]
        vector = self.X[:,7:9]
        speed = self.X[:,2:3]
        loc[...] += vector * ((speed + n) * timestep * self._dt)
        update_vector = np.linalg.norm(self.X[:,0:2] - self.X[:,3:5], axis=1) \
            > self.X[:,9]

        for i in np.where(update_vector)[0]:
            a = (self.X[i,3], self.X[i,4])
            b = (self.X[i,5], self.X[i,6])

            dest_list = list(self._roadmap.graph[b].keys())
            dest_list.remove(a)
            a = self.X[i,5:7]
            b = np.array(random.choice(dest_list))
            vector = b - a
            distance = np.linalg.norm(vector)
            vector = vector/distance
            self.X[i,0:10] = [a[0], a[1], self._v0, a[0], a[1], b[0],
                           b[1], vector[0], vector[1], distance]

    def update(self, z, R, p_fa=None, silent=False):
        data = {}
        data['Distance Before'] = z[0] - np.copy(self.X[:,0])
        if not silent: print("Examine weighting process")
        weight_addon = self._P_fa/self._roadmap.total_length
        if not silent: print('weight add on', weight_addon)
        w = (1. - self._P_fa)*mvn.pdf(self.X[:,0:2], z, R) + weight_addon
        data['0 raw'] = np.copy(w)
        if not silent: print('weight', [(w[idx], self.X[idx,0:2]-z) for idx in range(w.shape[0])])

        w = np.log(w)
        data['1 Log'] = np.copy(w)
        if not silent: print('log weight', w)
        max_w = np.max(w)
        if not silent: print('max weight', max_w)
        w = np.exp(w-max_w)
        data['2 Exponential'] = np.copy(w)
        if not silent: print('exp weight', w)
        # for code simplicity, normalize the weights here
        w = w/np.sum(w)
        data['3 Normalized'] = np.copy(w)
        if not silent: print('normalized weight', w)

        self.best_idx = np.argmax(w)
        self.best = self.X[self.best_idx]
        self.X[:,11] = w

        unique = self.low_var_sample()
        data['Distance After'] = z[0] - np.copy(self.X[:,0])
#         print('data 1 test: ', data)
        return data

    def neg_update(self, z, radius):
        self.X[:,11] = 1
        update_vector = np.linalg.norm(self.X[:,0:2] - z, axis=1) < radius
        self.X[update_vector,11] = self._P_miss
        self.X[:,11] /= np.sum(self.X[:,11])
        unique = self.low_var_sample()

class Particle(object):
    def __init__(self, roadmap, v0, dt, e0=None, x0=None, sigma=0.1):
        """A Particle contains the state and dynamic model of one hypothesis of a vehicle location.

        The particle's state consists of which road segment the vehicle is on and how far along
        that road segment the vehicle is, on a scale of 0 to 1. The particle also stores its
        nominal velocity and noise characteristics.

        roadmap -- object containing a graph describing the network of roads

        """
        # current edge
        self._roadmap = roadmap
        if e0 is None:
            a = random.choice(list(self._roadmap.graph.keys()))
            b = random.choice(list(self._roadmap.graph[a].keys()))
            self._e = (a, b)
        else:
            self._e = e0
        print(self._e)
        print(self._roadmap.graph.keys())
        print(self._roadmap.graph[self._e[0]].keys())
        self._e_len = self._roadmap.graph[self._e[0]][self._e[1]]
        # current position on edge
        if x0 is None:
            self._x = random.random()
        else:
            self._x = x0
        self._v = v0
        self._sigma = sigma
        self._dt = dt
        self.split = 0

    def predict(self):
        """Propogate the particle's state based on its dynamics and the roadmap

        When a particle is updated, it moves along the road segment by v0*dt, normalized by the
        length of its current road. If it reaches the end of the road (x >= 1), it queries the
        roadmap for other roads that connect to the current intersection and chooses one at
        random.
        """
        n = 0#np.random.normal(scale=self._sigma)
        self._x += (self._v + n)*self._dt/self._e_len
        self.split = 0

        if self._x >= 1.:
            dest_list = list(self._roadmap.graph[self._e[1]].keys())
            # no U-turns
            dest_list.remove(self._e[0])
            self.split = len(dest_list)
            self._e = (self._e[1], random.choice(dest_list))
            self._e_len = self._roadmap.graph[self._e[0]][self._e[1]]
            self._x -= 1

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

def get_vel_var(X):
    return np.var(X[:,:,2:3])

def getProb(variance, mean, pos):
    p = np.multiply((1/np.sqrt(2*np.pi*variance)), np.exp(-(pos-mean)/(2*variance)))
    return p

def getPredictedEntropy(variance, mean, max_distance, segment_size):
    entropy = 0
    for pos in range(0, max_distance, segment_size):
#         print(pos)
        p = getProb(variance, mean, pos)
        entropy -= p*np.log(p)
    return entropy

def get_pos_var(X, bins):
    return max(np.var(X[:,0:1]), np.var(X[:,1:2]))

def calc_entropy(r, X, res=5):
    """Returns the entropy of the estimate in nats

        r -- roadmap graph on which the particles exist
        X -- state of each particle, shape=(M, N, 12),
             M = number of targets
             N = number of particles
        """

    ## x, y, speed, start_x, start_y, end_x, end_y, direction_x, direction_y, distance, sigma, w
    M = X.shape[0]
    N = X.shape[1]
    # calculate the distance of each particle from the beginning of its road segment
    dists = np.linalg.norm(X[:, :, :2] - X[:, :, 3:5], axis=-1)
    dists_reverse = np.linalg.norm(X[:, :, :2] - X[:, :, 5:6], axis=-1)

    h = 0
    hist = []
    nodes_visited = []
    for start in r.keys():
#         if start not in nodes_visited:
#             nodes_visited.append(start)
            for end in r[start].keys():
#                 if end not in nodes_visited:
#                     nodes_visited.append(end)
                length = r[start][end]
                bin_start = 0.0
#                 bin_start_reverse = 1.0
                # find the particles on this road segment
                on_edge = np.all(X[:, :, 3:7] == start + end, axis=-1)
#                 on_edge_reverse = np.all(np.flip(X[:, :, 3:7], axis=2) == end + start, axis=-1)
                while bin_start < length:
                    in_bin = np.all([dists >= bin_start, dists <= bin_start + res], axis=0)
#                     in_bin_reverse = np.all([dists_reverse >= bin_start, dists_reverse <= bin_start + res], axis=0)

                    count = np.sum(np.all([on_edge, in_bin], axis=0))# + np.sum(np.all([on_edge, in_bin_reverse], axis=0))
                    p = count / (N*M)
                    hist.append(p)
                    if p > 0:
                        h -= p*np.log(p)
                    bin_start += res
#     get_vel_var(X)
    return h, get_pos_var(X, hist), get_vel_var(X)

def createGridLayout(x,y,min_edge_length, max_edge_length):
    nodes = []
    edges = []
    for i in range(y):
        for j in range(x):
            x_val = 0
            y_val = 0
            if i > 0:
                y_val = nodes[(i-1)*x + j][1]+ np.random.uniform(low=min_edge_length, high=max_edge_length)
            if j > 0:
                x_val = nodes[i*x+j-1][0]+ np.random.uniform(low=min_edge_length, high=max_edge_length)
            nodes.append((
                x_val,
                y_val ))

    for i in range(y):
        for j in range(x-1):
            edges.append((j+x*i,j+1+x*i))

    for i in range(y-1):
        for j in range(x):
            edges.append((j+x*i,j+x*(i+1)))
    return [nodes, edges]

class AgentSimple(object):
    def __init__(
        self, center, width, height, speed, period,
        roadmap, dt, fov=30.):
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
        self.discount = .8

    def update(self, pfs):
#         ##Dummy version of path planning
#         self._t += self._dt
#         self.pos = np.array((self._w*np.cos(2*np.pi*self._t/self._period) + self._c[0],
#                              self._h*np.sin(2*2*np.pi*self._t/self._period) + self._c[1]))
#         if self._sc_agent is not None:
#             self.update_plot()

        ##inteligent path planning
        self._t += self._dt
        distance = np.array((
            self.path[self.current_dest][0] - self.pos[0],
            self.path[self.current_dest][1] - self.pos[1]))

        if abs(distance[0]) > abs(distance[1]):
            distance[1] *= 20
        elif abs(distance[1]) > abs(distance[0]):
            distance[0]*= 20

        vel = self.speed * (distance / np.linalg.norm(distance) )
        self.pos = self.pos + np.array(vel * self._dt)

        if self._sc_agent is not None:
            self.update_plot()
        if np.linalg.norm(distance) < 5:
            self.update_path(pfs)

    def update_path(self, pfs):
        destinations = list(self._roadmap.graph[self.path[self.current_dest]].keys())
        if len(self.path) > 1:
            destinations.remove(self.path[self.current_dest-1])
        self.path.append(random.choice(destinations))
        self.current_dest += 1

    def init_plot(self, ax):
        self._sc_agent = ax.scatter([self.pos[0]], [self.pos[1]], s=50, marker='D', facecolor='red', label='agent')
        self._sc_fov = plt.Circle((self.pos[0], self.pos[1]), self.fov, facecolor='None', edgecolor='orange')
        ax.add_patch(self._sc_fov)

    def update_plot(self):
        self._sc_agent.set_offsets([self.pos])
        self._sc_fov.center = self.pos[0], self.pos[1]

def sim(r, pf, targets, agent, R, dt, T_end, num_sightings, plot=True):
    def grow_loop(iterations, pf, targets, H, variances):
        for i in range(iterations):
            pf.predict()
            dists = []
            for target in targets:
                target.predict()
            if plot:
                locs1 = pf.X[:,:2]
                sc1.set_offsets(locs1)
                sc_target1.set_offsets(targets[0].loc)

            if plot:
                fig.canvas.draw()

            X = np.array([pf.X])
            H_current, var_pos, var_vel = calc_entropy(r.graph, X)
            H += [H_current]
            variances += [var_pos]
        return H, variances

    def positive_sighting(targets, R, agent, pf, r, H, variances):
        for target in targets:
            z = target.loc
            data = pf.update(z,R,silent=True)
            X = np.array([pf.X])
            H_current, var_pos, var_vel = calc_entropy(r.graph, X)
            H += [H_current]
            variances += [var_pos]
            return H, variances, data

    if plot:
        fig, ax = plt.subplots()
        r.visualize(ax)

        x0 = rbpf.best[0].X[:,0]
        y0 = rbpf.best[0].X[:,1]
        sc1 = ax.scatter(x0, y0, s=10, linewidth=0, facecolor='green', label='particles')
        loc = r.get_loc(targets[0].state)
        sc_target1 = ax.scatter([loc[0]], [loc[1]], s=100, marker='*', facecolor='green', label='target')

        ax.legend()

        ax.set_aspect('equal')

        fig.canvas.draw()
        start = time.time()
        tic = start
    H = []
    Ts = dt

    variances = []
    for i in range(15):
        pf.predict()
        targets[0].predict()
        pf.update(mvn.rvs(targets[0].loc, R), R, silent=True)

    update_data = []

    for idx in range(num_sightings):
        T_end_tmp = int(T_end/(2**(idx+1))/Ts)
        H, variances = grow_loop(T_end_tmp, pf, targets, H, variances)
        H, variances, update_info = positive_sighting(targets, R, agent, pf, r, H, variances)
        update_data.append(update_info)

    return H, variances, update_data

def getVariancesParallel(x,y,dist_e,N, dt, T_end, num_runs, P_fa=0.0,P_miss=0.0, sigma_vel=4, Va=30, v0=0, R_val=2):
    R = R_val*np.eye(2)
    nodes, edges = createGridLayout(x,y,dist_e,dist_e)
    r = Roadmap(nodes, edges)
    targets = [Particle(r,v0=v0,dt=dt,e0=((0,0),(dist_e,0)), x0=.5, sigma=sigma_vel)]
    pf_args = {'roadmap':r, 'num_particles':N, 'dt':dt, 'v0':v0,
               'sigma':sigma_vel, 'P_fa':P_fa, 'P_miss':P_miss}
    pf = PF(**pf_args)
    agent = AgentSimple((dist_e/2,0), 100, 50, Va, 30, r, dt=dt)
    with Pool(processes=num_runs) as pool:
        data = pool.starmap(
                sim, [[r, pf, targets, agent, R, dt, T_end, 1,
                False] for idx in range(num_runs)])
        data = np.array(data)
        data_h = data[:,0]
        data_v = data[:,1]
        data_u = data[:,2]
        np.save("variance_{}x{}-{}-{}_h".format(x,y,dist_e, R_val), data_h)
        np.save("variance_{}x{}-{}-{}_v".format(x,y,dist_e, R_val), data_v)
        np.save("variance_{}x{}-{}-{}_u".format(x,y,dist_e, R_val), data_u)

if len(sys.argv) == 5:
    dist_e = float(sys.argv[1])
    entropy_resolution = int(sys.argv[2])
    x = int(sys.argv[3])
    y = int(sys.argv[4])
else:
    dist_e = 50.
    entropy_resolution = 1
P_fa = 0.0
P_miss = 0.0
v0 = 10
sigma = 4
map_size = dist_e*7


base = {
    0: {'offset': dist_e/v0,
       'options': [1,1]},
    1: {'offset': dist_e*3/v0,
       'options': [0,1]}
}
N = 100000
dt = 0.1
T_end = 120/2
num_runs = 1
fov = 30
sigma_vel = 6
Va = 30

# h, v = runIterations(nodes, edges, dist_e, N, dt, T_end, num_runs, P_fa, P_miss, fov=fov,
#                   v0=v0, sigma=sigma, num_targets=2, plot=False, agent_active=True, entropy_resolution=entropy_resolution)

# runParallelIterations(x, y, dist_e, N, dt, T_end, num_runs, P_fa,
    # P_miss, fov=fov, v0=v0, sigma=sigma, num_targets=2, plot=False,
    # agent_active=True, entropy_resolution=entropy_resolution)
getVariancesParallel(x,y,dist_e,N,dt,T_end,num_runs,P_fa,P_miss,sigma_vel,Va,v0)

##### from __future__ import division

# %matplotlib notebook
#%matplotlib inline
import sys
import gc
import time
import random
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from math import atan2, log, ceil
from tqdm import tqdm
from tqdm import trange
import copy
import datetime
from collections import Counter
from matplotlib import colors as mcolors

from importlib import reload
import roadmap
reload(roadmap)
from roadmap import Roadmap
import particle
reload(particle)
from particle import Particle
import pf
reload(pf)
from pf import PFJupyter as PF
import agent
reload(agent)
from agent import AgentJupyterPerfect as AgentPerfect

def calc_entropy(graph, X, res=1):
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
    dists = np.linalg.norm(X[:,:, :2] - X[:,:, 3:5], axis=-1)

    h = np.zeros(M)
    for start in graph.keys():
            for end in graph[start].keys():
                length = graph[start][end]
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
    return h#, [get_pos_var(X[idx], hist) for idx in range(M)], counts, probs

def getProb(variance, mean, pos):
    p = np.multiply((1/np.sqrt(2*np.pi*variance)), np.exp(-(pos-mean)/(2*variance)))
    return p

def getPredictedEntropy(variance, mean, max_distance, segment_size):
    entropy = 0
    for pos in range(0, max_distance, segment_size):
        p = getProb(variance, mean, pos)
        entropy -= p*np.log(p)
    return entropy

class RB_PF(object):
    def __init__(self, roadmap, num_particles, max_vehicles, pf_args):
        self._roadmap = roadmap
        self._N = num_particles
        self._max_vehicles = max_vehicles
        self.X = [[PF(**pf_args) for j in range(self._max_vehicles)] for i in range(self._N)]
        self.best = self.X[0]
        self.no_measurements = True

    def lowVarSample(self, w):
        Xbar = []
        M = self._N
        r = np.random.uniform(0, 1/M)
        c = w[0]
        i = 0
        last_i = i
        unique = 1
        for m in range(M):
            u = r + m/M
            while u > c:
                i += 1
                c = c + w[i]
            new_x = copy.deepcopy(self.X[i])
            Xbar.append(new_x)
            if i == self.best_idx:
                self.best = new_x
            if last_i != i:
                unique += 1
            last_i = i
        self.X = Xbar
        return unique

    def predict(self):
        # propagate each bank of particle filters
        [[xi.predict() for xi in x] for x in self.X]

    def update(self, z, R, lone_target, radius=None, p_fa=None):
        w = np.zeros(self._N)

        for i, x in enumerate(self.X):
            if self.no_measurements:
                t = 0
            else:
                # get the likelihood that the measurement came from each target
                l = np.array([xi.get_measurement_likelihood(z, R) for xi in x])

                # normalize the likelihoods so we can randomly choose a corresponding target
                # with some smart probabilites
                if np.sum(l) < 10*np.finfo(float).eps:
                    continue
                l = l/np.sum(l)
                t = np.where(np.random.multinomial(1, l) == 1)[0][0]

            w[i] = x[t].get_measurement_likelihood(z, R)
            x[t].update(z, R)
            if lone_target:
                for j, xi in enumerate(x):
                    if t != j:
                        xi.neg_update(z, radius)
        self.no_measurements = False


        # logsumexp
        max_w = np.max(w)
        w = np.exp(w-max_w)
        # for code simplicity, normalize the weights here
        w = w/np.sum(w)
#         print("best: {}={}".format(np.argmax(w), np.max(w)))

        self.best_idx = np.argmax(w)
        self.best = self.X[self.best_idx]
        unique = self.lowVarSample(w)
#         print(unique)


    def neg_update(self, z, radius):
        [[xi.neg_update(z, radius) for xi in x] for x in self.X]

def sim(rbpf, r, targets, R, dt, T_end, plot=True, agent=None, unknown_start=False):
    H = []
    Ts = dt
    freq = 6#int(1/Ts/2)

    # update the scenario
    for i in range(int(T_end/Ts)):
        for pf in rbpf:
            pf.predict()
        dists = []
        for target in targets:
            target.predict()
            if agent is not None:
                dists.append(np.linalg.norm(target.loc - agent.pos))
        if agent is not None:
            if agent is not None:
                if i % freq == 0 and i != 0:
                    if (dists[0] < agent.fov) != (dists[1] < agent.fov):
                        if dists[0] < agent.fov:
                            z = mvn.rvs(targets[0].loc, R)
                            rbpf[0].update(z,R)
                        if dists[1] < agent.fov:
                            z = mvn.rvs(targets[1].loc, R)
                            rbpf[1].update(z,R)
                    elif (dists[0] < agent.fov) and (dists[1] < agent.fov):
                            z = mvn.rvs(targets[0].loc, R)
                            rbpf[0].update(z,R)
                            z = mvn.rvs(targets[1].loc, R)
                            rbpf[1].update(z,R)
        pfs = rbpf
        if agent is not None:
            agent.update(pfs, targets)
        X = np.array([rbpf[i].X for i in range(len(rbpf))])
        H_current = calc_entropy(r.graph, X)
        H += [H_current]
    return H

def runIterations(nodes, edges, N, dt, T_end, num_runs, P_fa=.02,
                  P_miss=.05, v0=4.5,
                  sigma=4, num_targets=1, e0=None, x0=None,
                  one_edge=False, plot=False, agent_path=None,
                 get_entropy=True, get_variance=False,
                  get_counts=False, get_prob=False,
                  get_particles=False,
                 unknown_start=False, name=None):
    r = Roadmap(nodes, edges, rotate=False)
    pf_args = {'roadmap':r, 'num_particles':N, 'dt':dt, 'v0':v0, 'sigma':sigma,
               'P_fa':P_fa, 'P_miss':P_miss, 'one_edge': one_edge}
    R = 5*np.eye(2)
    H = []
    for idx in trange(num_runs):
        gc.collect()
        rbpf = [PF(**pf_args) for j in range(num_targets)]
        targets = []
        for jdx in range(num_targets):
            if x0 is not None:
                targets.append(Particle(r, v0=v0, dt=dt,e0=e0[jdx], x0=x0[jdx], sigma=sigma))
            else:
                targets.append(Particle(r, v0=v0, dt=dt, sigma=sigma))

        if agent_path is None:
            agent = None
        else:
            agent = AgentPerfect(40, r, dt=dt)
        H_tmp = sim(rbpf, r, targets, R, dt, T_end, plot=plot, agent=agent, unknown_start=unknown_start)
        if get_entropy:
            H += [H_tmp]
            if idx % 10 == 0:
                np.save('entropy_temp_{}'.format(name), H)
        rbpf = None
        targets = None
        H_tmp = None
        agent = None
    return_data = []
    if get_entropy:
        return_data.append(H)
    return return_data

def getEntropy(probability, mode_probability):
    entropy = 0
    for num_modes in mode_probability.keys():
        for idx in range(len(probability)):
            if np.isfinite(np.log((1/num_modes)*probability[idx])) and probability[idx] > 0:
                entropy += mode_probability[num_modes]*(-probability[idx]*np.log((1/num_modes)*probability[idx]))
    return entropy
def getProbAndEntropy(variance, x, mu, mode_probability,step_size):
    probability = 1./np.sqrt(2*np.pi*variance)*np.exp(-(x-mu)**2/(2*variance))
    cdf = []
    for idx in range(0,len(probability),step_size):
        cdf.append(sum(probability[:idx]))
    probability = cdf - np.concatenate(([0], cdf[0:-1]))
    entropy = getEntropy(probability, mode_probability)
    return probability, entropy
def getProbAndEntropyOverTime(var_0, map_size, dt, sigma, num_steps, mode_probability, step_size=1):
    mu = map_size/2.
    distance = np.array([idx for idx in range(0,int(map_size), 1)])
    prob_over_time = []
    entropy_over_time = []
    for idx in range(num_steps):
        var = var_0 + idx*(dt*sigma)**2
        prob, entropy = getProbAndEntropy(var, distance, mu, mode_probability[idx], step_size)
        prob_over_time.append(prob)
        entropy_over_time.append(entropy)
    return prob_over_time, entropy_over_time
def get_time(t, ts, offset, initial_range):
    time = t - (ts + offset)
    time[np.where(time<0)] = 0
    time[np.where(time>initial_range)] = initial_range
    return time
def get_prob_of_mode_type(prob, initial_range, ts, options, base, t, level, desired_level):
    z = np.zeros((t.shape[0]))
    o = np.zeros((t.shape[0]))
    p = z
    num_options = len(options)
    next_prob = 1./num_options
    if level == desired_level:
        p = prob*get_time(t, ts, 0, initial_range)
        for idx in range(num_options):
            offset = base[options[idx]]['offset']
            p = p - prob*next_prob*get_time(t, ts, offset, initial_range)
    else:
        first_death = t[-1]+1
        if level < desired_level:
            for idx in range(num_options):
                offset = base[options[idx]]['offset']
                next_options = base[options[idx]]['options']
                p = p + prob*get_prob_of_mode_type(next_prob, initial_range, ts+offset, next_options, base, t, level+1, desired_level)
    return p
def getModeProbabilities(prob, initial_range, base, t, max_level):
    probabilities = {}
    start_level = 2
    prob_sum = [1 for idx in t]
    for desired_level in range(start_level,max_level):
        probabilities[desired_level] = [0 for idx in t]
        for idx in range(len(prob)):
            probabilities[desired_level] += prob[idx]*(1./initial_range[idx])*get_prob_of_mode_type(
                1, initial_range[idx], 0, base[idx]['options'], base, t, start_level, desired_level)
        prob_sum -= probabilities[desired_level]
    probabilities[1] = prob_sum
    return probabilities

## Test Perfect Agent
dist_e = 100
nodes = [(0.,0.), (dist_e,0.), (2*dist_e,0.),
        (0.,dist_e,),(dist_e,dist_e),(2*dist_e,dist_e),
        (0.,2*dist_e),(dist_e,2*dist_e),(2*dist_e,2*dist_e)]

edges = [(0, 1),(1,2),(0,3),(1,4),(2,5),(3,4),(4,5),(3,6),(4,7),(5,8),(6,7),(7,8)]

name = sys.argv[1]

N = 1000
dt = 0.1
T_end = 150
num_runs = 500
Va=40

v0 = 10
sigma = 4

h, v, p = runIterations(nodes, edges, N, dt, T_end, num_runs, v0=v0,
                           sigma=sigma, num_targets=2,one_edge=True,
                        agent_path=True, P_fa=0.0, P_miss=0.0,
                           get_entropy=True, get_variance=True,
                        get_prob=True, plot=False, unknown_start=True, name=name)

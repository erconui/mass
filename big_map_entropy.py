from importlib import reload
import numpy as np
import time
from multiprocessing import Pool

from scipy.stats import multivariate_normal as mvn
import copy
import matplotlib.pyplot as plt
from math import atan2, log, ceil
import os
import roadmap
reload(roadmap)
from roadmap import Roadmap
import pf
reload(pf)
from pf import PFJupyter as PF
import particle
reload(particle)
from particle import Particle

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

def calc_entropy(graph, X, res=1, entropy_only=True):
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
#         print(M)
    # calculate the distance of each particle from the beginning of its road segment
    dists = np.linalg.norm(X[:,:, :2] - X[:,:, 3:5], axis=-1)
#         dists_reverse = np.linalg.norm(X[:,:, :2] - X[:, 5:6], axis=-1)

    h = np.zeros(M)
    hist = []
    nodes_visited = []
    counts = []
    probs = []
    for start in graph.keys():
#         if start not in nodes_visited:
#             nodes_visited.append(start)
            for end in graph[start].keys():
#                 if end not in nodes_visited:
#                     nodes_visited.append(end)
                length = graph[start][end]
#                 bin_start_reverse = 1.0
                # find the particles on this road segment
                on_edge = np.all(X[:,:, 3:7] == start + end, axis=-1)
#                 on_edge_reverse = np.all(np.flip(X[:, :, 3:7], axis=2) == end + start, axis=-1)
                for idx in range(M):
                    bin_start = 0.0
                    while bin_start < length:
#                             if idx != 0:
#                                 print('idx', idx)
                        in_bin = np.all([dists >= bin_start, dists <= bin_start + res], axis=0)
    #                     in_bin_reverse = np.all([dists_reverse >= bin_start, dists_reverse <= bin_start + res], axis=0)

                        count = np.sum(np.all([on_edge[idx], in_bin[idx]], axis=0))# + np.sum(np.all([on_edge, in_bin_reverse], axis=0))
#                             if idx > 0:
#                             if idx != 0:
#                                 print('idx', idx, 'count',count)
                        p = count / (N)
                        if not entropy_only:
                            counts.append(count)
                            probs.append(p)
                            hist.append(p)
                        if p > 0:
                            h[idx] -= p*np.log(p)
                        bin_start += res
#     print(np.linalg.norm(h),)
    if entropy_only:
        return h
    return h, [get_pos_var(X[idx], hist) for idx in range(M)], counts, probs

def get_pos_var(X, bins):
    return max(np.var(X[:,:,0:1]), np.var(X[:,:,1:2]))

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

def sim(rbpf, r, targets, R, dt, T_end, plot=True, agent=None):
    if plot:
        fig, ax = plt.subplots()
        r.visualize(ax)

        x0 = rbpf.best[0].X[:,0]
        y0 = rbpf.best[0].X[:,1]
        sc1 = ax.scatter(x0, y0, s=10, linewidth=0, facecolor='green', label='particles')
        loc = r.get_loc(targets[0].state)
        sc_target1 = ax.scatter([loc[0]], [loc[1]], s=100, marker='*', facecolor='green', label='target')
        if agent is not None:
            agent.init_plot(ax)
        ax.legend()

        ax.set_xlim(-20, 450)
        ax.set_ylim(-20, 320)
        ax.set_aspect('equal')
        fig.canvas.draw()
        start = time.time()
        tic = start
    H = []
    Ts = dt
    avg_distance = [[] for pf in rbpf.best]
    num_in_threshold = [[] for pf in rbpf.best]
    variances = []
    counts = []
    probs = []
    particles = []
    for i in range(15):
        targets[0].predict()
        rbpf.predict()
        rbpf.update(mvn.rvs(targets[0].loc, R), R, lone_target=True, radius=1)

    # update the scenario
    for i in range(int(T_end/Ts)):
        rbpf.predict()
        dists = []
        for target in targets:
            target.predict()
            if agent is not None:
                dists.append(np.linalg.norm(target.loc - agent.pos))
        if agent is not None:
            if agent is not None:
                if i % int(1/Ts) == 0 and i != 0:
                    if dists[0] < agent.fov:
                        z = mvn.rvs(targets[0].loc, R)
                        rbpf.update(z, R, lone_target=True, radius=agent.fov*0.75)
                    else:
                        rbpf.neg_update(agent.pos, radius=agent.fov*0.75)
        if plot:
            locs1 = rbpf.best[0].X[:,:2]
            sc1.set_offsets(locs1)
            sc_target1.set_offsets(targets[0].loc)
            pfs = rbpf.best
        pfs = rbpf.best
        if agent is not None:
            agent.update(None)
        if plot:
            fig.canvas.draw()
            toc = time.time()
            dur = toc - tic
            tic = toc
        X = np.array([rbpf.best[i].X for i in range(len(rbpf.best))])
        # H_current, var_pos, part_dist, p = calc_entropy(r.graph, X)
        H_current = calc_entropy(r.graph, X)
        H += [H_current]
        # variances.append(var_pos)
        # counts += [part_dist]
        # probs += [p]
        # particles += [X]

    return H#, variances, counts, probs, particles


def runSim(r, R, N, dt, P_fa=.02, P_miss=.05, fov=30, v0=4.5, Va=40, sigma=4, num_targets=1, e0=None, x0=None, one_edge=False, plot=False, agent_active=False, all_data=False, entropy_resolution=1):
    pf_args = {'roadmap':r, 'num_particles':N, 'dt':dt, 'v0':v0, 'sigma':sigma,
               'P_fa':P_fa, 'P_miss':P_miss, 'one_edge': one_edge}
    rbpf = RB_PF(r, 10, num_targets, pf_args)
    targets = []
    for jdx in range(num_targets):
        if x0 is not None:
            targets.append(Particle(r, v0=v0, dt=dt,e0=e0[jdx], x0=x0[jdx], sigma=sigma))
        else:
            targets.append(Particle(r, v0=v0, dt=dt, sigma=sigma))
    # if agent_active is None:
    agent = None
    # else:
        # agent = AgentPerfect((100, 60), 100, 50, 40, 30, r, dt=dt, path=agent_path)
    if all_data:
        return sim(rbpf, r, targets, R, dt, T_end, plot=plot, agent=agent)
    else:
        return sim(rbpf, r, targets, R, dt, T_end, plot=plot, agent=agent)


def runIterations(nodes, edges, N, dt, T_end, num_runs, P_fa=.02, P_miss=.05, v0=4.5, sigma=4, num_targets=1, e0=None, x0=None, one_edge=False, plot=False, agent_active=None):
    r = Roadmap(nodes, edges, rotate=False)
    pf_args = {'roadmap':r, 'num_particles':N, 'dt':dt, 'v0':v0, 'sigma':sigma,
               'P_fa':P_fa, 'P_miss':P_miss, 'one_edge': one_edge}
    R = 5*np.eye(2)
    H = []
    for idx in trange(num_runs):
        # H_tmp, variance, count, prob, part = runSim(r, R, N, dt, P_fa=P_fa, P_miss=P_miss, v0=v0, sigma=sigma, num_targets=num_targets, e0=e0, x0=x0, one_edge=one_edge, plot=plot, agent_active=agent_active)
        H_tmp = runSim(r, R, N, dt, P_fa=P_fa, P_miss=P_miss, v0=v0, sigma=sigma, num_targets=num_targets, e0=e0, x0=x0, one_edge=one_edge, plot=plot, agent_active=agent_active)
        H += [H_tmp]
    return_data = [np.nanmean(H, axis=0)]
    return return_data

def runParallelIterations(x, y, dist_e, N, dt, T_end, num_runs, P_fa=.02, P_miss=.05, fov=30, v0=4.5, Va=40, sigma=4, num_targets=1, e0=None, x0=None, one_edge=False, plot=False, agent_active=None, all_data=False, entropy_resolution=1):
    nodes, edges = createGridLayout(x,y,dist_e,dist_e)
    r = Roadmap(nodes, edges, rotate=False)
    pf_args = {'roadmap':r, 'num_particles':N, 'dt':dt, 'v0':v0, 'sigma':sigma,
               'P_fa':P_fa, 'P_miss':P_miss, 'one_edge': one_edge}
    R = 5*np.eye(2)
    H = []

    with Pool(processes=num_runs) as pool:
        if all_data:
            H_tmp, variance, count, prob, part = runSim(r, R, N, dt, P_fa=P_fa,
                P_miss=P_miss, v0=v0, Va=Va, sigma=sigma, num_targets=num_targets,
                e0=e0, x0=x0, one_edge=one_edge, plot=plot,
                agent_active=agent_active)
        else:
            data = pool.starmap(runSim, [[r, R, N, dt, P_fa,
                P_miss, fov, v0, Va, sigma, num_targets,
                e0, x0, one_edge, plot,
                agent_active, all_data, entropy_resolution] for idx in range(num_runs)])
            data = np.array(data)
            data_h = data[:,0]
            np.save("data_{}x{}-{}-{}_h".format(x,y,int(dist_e), entropy_resolution), data_h)



dist_e = 1
dim = 30
nodes, edges = createGridLayout(dim,dim,dist_e, dist_e)
dt = .1
v0=.1
Va = .4
sigma = .04
N = 100000
T_end = 179
num_runs = 5
# dur = getDuration(5,5, dist_e, Va)
# print(dur)
# h = runIterations(nodes, edges, N, dt, T_end, num_runs, v0=v0,
#                            sigma=sigma, e0=[(nodes[1], nodes[0])], x0=[.03], one_edge=True,
#                            plot=False)
# np.save('BigMapEntropy', h)
runParallelIterations(30, 30, dist_e, N, dt, T_end, num_runs, P_fa=.02,
                      P_miss=.05, fov=30, v0=4.5, Va=40, sigma=4, num_targets=1,
                      e0=None, x0=None, one_edge=False, plot=False,
                      agent_active=False, all_data=False, entropy_resolution=1)

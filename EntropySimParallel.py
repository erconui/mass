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
from particle import Particle
from roadmap import Roadmap
# import pf
# reload(pf)
from pf import PFJupyter as PF
# import agent
# reload(agent)
# from agent import AgentJupyter as Agent
# from agent import AgentJupyterRandom as AgentRandom
from agent import AgentJupyterPerfect as AgentPerfect

# Variance and Entropy calculations
def get_vel_var(X):
    return np.var(X[:,:,2:3])

def getProb(variance, mean, pos):
    p = np.multiply((1/np.sqrt(2*np.pi*variance)), np.exp(-(pos-mean)/(2*variance)))
    return p

def getPredictedEntropy(variance, mean, max_distance, segment_size):
    entropy = 0
    for pos in range(0, max_distance, segment_size):

        p = getProb(variance, mean, pos)
        entropy -= p*np.log(p)
    return entropy

def get_pos_var(X, bins):
    return max(np.var(X[:,0:1]), np.var(X[:,1:2]))

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
#         print(M)
    # calculate the distance of each particle from the beginning of its road segment
    dists = np.linalg.norm(X[:,:, :2] - X[:,:, 3:5], axis=-1)

    h = np.zeros(M)
    # hist = []
    nodes_visited = []
    # counts = []
    # probs = []
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
                    # counts.append(count)
                    p = count / (N)
                    # probs.append(p)
                    # hist.append(p)
                    if p > 0:
                        h[idx] -= p*np.log(p)
                    bin_start += res
    return h#, counts, probs


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

def sim(rbpf, r, targets, R, dt, T_end, plot=True, agent=None, unknown_start=True, entropy_resolution=1):
    if plot:
        fig, ax = plt.subplots()
        r.visualize(ax)

#         x0 = rbpf.best[0].X[:,0]
#         y0 = rbpf.best[0].X[:,1]
        x0 = rbpf[0].X[:,0]
        y0 = rbpf[0].X[:,1]
        sc1 = ax.scatter(x0, y0, s=10, linewidth=0, facecolor='green', label='particles')
        loc = r.get_loc(targets[0].state)
        sc_target1 = ax.scatter([loc[0]], [loc[1]], s=100, marker='*', facecolor='green', label='target')

#         x0 = rbpf.best[1].X[:,0]
#         y0 = rbpf.best[1].X[:,1]
        x0 = rbpf[1].X[:,0]
        y0 = rbpf[1].X[:,1]
        sc2 = ax.scatter(x0, y0, s=10, linewidth=0, facecolor='blue', label='particles')
        loc = r.get_loc(targets[1].state)
        sc_target2 = ax.scatter([loc[0]], [loc[1]], s=100, marker='*', facecolor='blue', label='target')
        if agent is not None:
            agent.init_plot(ax)
        ax.legend()
        #ax.plot([50, 60], [50, 50], marker='o', ls='None')

        ax.set_xlim(-20, 450)
        ax.set_ylim(-20, 320)
        ax.set_aspect('equal')
        # plt.plot([0,1,1,2])
        # plt.show()
        fig.canvas.draw()
        start = time.time()
        tic = start
    H = []
    Ts = dt
#     avg_distance = [[] for pf in rbpf.best]
#     num_in_threshold = [[] for pf in rbpf.best]
    avg_distance = [[] for pf in rbpf]
    num_in_threshold = [[] for pf in rbpf]
    variances = []
    counts = []
    probs = []
    particles = []
    freq = 4
    if not unknown_start:
        for i in range(30):
            targets[0].predict()
            rbpf.predict()
            rbpf.update(mvn.rvs(targets[0].loc, R), R, lone_target=True, radius=1)

    # update the scenario
    for i in range(int(T_end/Ts)):
#    for i in tqdm(range(int(T_end/Ts))):
#         rbpf.predict()
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
    #                         rbpf.update(z, R, lone_target=True, radius=agent.fov*0.75)
                            rbpf[0].update(z,R)
    #                         rbpf[1].neg_update(z,agent.fov*0.75)
                        if dists[1] < agent.fov:
                            z = mvn.rvs(targets[1].loc, R)
    #                         rbpf.update(z, R, lone_target=True, radius=agent.fov*0.75)
                            rbpf[1].update(z,R)
    #                         rbpf[0].neg_update(z,agent.fov*0.75)
                    elif (dists[0] < agent.fov) and (dists[1] < agent.fov):
                            z = mvn.rvs(targets[0].loc, R)
    #                         rbpf.update(z, R, lone_target=False)
                            rbpf[0].update(z,R)
                            z = mvn.rvs(targets[1].loc, R)
    #                         rbpf.update(z, R, lone_target=False)
                            rbpf[1].update(z,R)
    #                 else:
    # #                     rbpf.neg_update(agent.pos, radius=agent.fov*0.75)
    #                         rbpf[0].neg_update(agent.pos,agent.fov*0.75)
    #                         rbpf[1].neg_update(agent.pos,agent.fov*0.75)
        if plot:
#             locs1 = rbpf.best[0].X[:,:2]
            locs1 = rbpf[0].X[:,:2]
            sc1.set_offsets(locs1)
#             locs2 = rbpf.best[1].X[:,:2]
            locs2 = rbpf[1].X[:,:2]
            sc2.set_offsets(locs2)
            sc_target1.set_offsets(targets[0].loc)
            sc_target2.set_offsets(targets[1].loc)
#             pfs = rbpf.best
            pfs = rbpf
#         pfs = rbpf.best
        pfs = rbpf
        if agent is not None:
            agent.update(pfs, targets)
        if plot:
            fig.canvas.draw()
            toc = time.time()
            dur = toc - tic
            tic = toc
#         X = np.array([rbpf.best[i].X for i in range(len(rbpf.best))])
        X = np.array([rbpf[i].X for i in range(len(rbpf))])
        # H_current, var_pos, part_dist, p = calc_entropy(r.graph, X)
        H_current = calc_entropy(r.graph, X)
        H += [H_current]
        # print(H)
        # variances.append(var_pos)
        # counts += [part_dist]
        # probs += [p]
        # particles += [X]
#         variances[1].append(var_vel)

    return H#, variances, counts, probs, particles

def runSim(r, R, N, dt, P_fa=.00, P_miss=.05, fov=30, v0=4.5, Va=40, sigma=4, num_targets=1, e0=None, x0=None, one_edge=False, plot=False, agent_active=False, all_data=False, entropy_resolution=1):
    # print(num_targets)
    pfs = [PF(r, N, dt, v0=v0, sigma=sigma, P_fa=P_fa, P_miss=P_miss)
           for idx in range(num_targets)]
    targets = []
    for jdx in range(num_targets):
        if x0 is not None:
            targets.append(Particle(r, v0=v0, dt=dt,e0=e0[jdx], x0=x0[jdx], sigma=sigma))
        else:
            targets.append(Particle(r, v0=v0, dt=dt, sigma=sigma))
    if agent_active:
        agent = AgentPerfect(Va, r, dt=dt, fov=fov, entropy_resolution=entropy_resolution)
    else:
        agent = None
    if all_data:
        return sim(pfs, r, targets, R, dt, T_end, plot=plot, agent=agent,
                                                 all_data=all_data, entropy_resolution=entropy_resolution)
    else:
        data =  sim(pfs, r, targets, R, dt, T_end, plot=plot, agent=agent, entropy_resolution=entropy_resolution)
        print(data)
        return data

def runParallelIterations(x, y, dist_e, N, dt, T_end, num_runs, R, P_fa=.02, P_miss=.05, fov=30, v0=4.5, Va=40, sigma=4, num_targets=1, e0=None, x0=None, one_edge=False, plot=False, agent_active=False, all_data=False, entropy_resolution=1, unit=None):
    nodes, edges = createGridLayout(x,y,dist_e,dist_e)
    r = Roadmap(nodes, edges, rotate=False)
    pf_args = {'roadmap':r, 'num_particles':N, 'dt':dt, 'v0':v0, 'sigma':sigma,
               'P_fa':P_fa, 'P_miss':P_miss, 'one_edge': one_edge}
    R = R*np.eye(2)
    H = []
    variances = []
    args = []

    with Pool(processes=num_runs) as pool:
        if all_data:
            H_tmp, variance, count, prob, part = runSim(r, R, N, dt, P_fa=P_fa,
                P_miss=P_miss, fov=fov, v0=v0, Va=Va, sigma=sigma,
                num_targets=num_targets, e0=e0, x0=x0, one_edge=one_edge,
                plot=plot, agent_active=agent_active, all_data=all_data,
                entropy_resolution=entropy_resolution)
        else:
            data = pool.starmap(runSim, [[r, R, N, dt, P_fa, P_miss,
                fov, v0, Va, sigma, num_targets,
                e0, x0, one_edge, plot,
                agent_active, all_data,
                entropy_resolution] for idx in range(num_runs)])
            data = np.array(data)
            print("returned")
            print(data)
            data_h = data#[:,0]
            # data_v = data[:,1]
            np.save("ideal_h_{}".format(unit), data_h)
            # np.save("data_{}x{}-{}-{}_v".format(x,y,int(dist_e), entropy_resolution), data_v)


if len(sys.argv) == 6:
    dist_e = float(sys.argv[1])
    entropy_resolution = int(sys.argv[2])
    x = int(sys.argv[3])
    y = int(sys.argv[4])
    unit = sys.argv[5]
else:
    dist_e = 50.
    entropy_resolution = 1
P_fa = 0.0
P_miss = 0.0
v0 = 10
sigma = 4
map_size = dist_e*7
R = 5

base = {
    0: {'offset': dist_e/v0,
       'options': [1,1]},
    1: {'offset': dist_e*3/v0,
       'options': [0,1]}
}
N = 1000
dt = 0.1
T_end = 200
num_runs = 10
fov = 30

runParallelIterations(x, y, dist_e, N, dt, T_end, num_runs, R, P_fa,
    P_miss, fov=fov, v0=v0, sigma=sigma, num_targets=2, plot=False,
    agent_active=True, entropy_resolution=entropy_resolution, unit=unit)

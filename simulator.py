from IPython.core.debugger import set_trace

from importlib import reload

import numpy as np
from scipy.stats import multivariate_normal as mvn
import random

from rbpf import RBPF
# from roadmap import Roadmap
from particle import Particle
import agent
reload(agent)
from agent import AgentJupyter

# import matplotlib
import matplotlib.pyplot as plt
# matplotlib.rcParams['figure.dpi'] = 100

class Simulator(object):

    def __init__(self, nodes, edges, roadmap, P_fa=.02, P_miss=.05, N=250, v0=10, sigma=4, num_targets=1):
        self.nodes = nodes
        self.edges = edges
        self.num_targets = num_targets
        #
        self.dt = .1
        # self.roadmap = Roadmap(self.nodes, self.edges, rotate=False)
        self.roadmap = roadmap

        self.fig = None

        self.R = 5*np.eye(2)

        self.P_fa = P_fa
        self.P_miss= P_miss
        self.N = N
        self.v0 = v0
        self.sigma = sigma
        self.reset()
    #
    def setup_visualization(self):
        # print('test')
        # plt.tight_layout()
        self.fig, ax = plt.subplots()#figsize=(10,10))
        # self.fig.
        self.roadmap.visualize(ax)
        x0 = self.rbpf.best[0].X[:,0]
        y0 = self.rbpf.best[0].X[:,1]
        self.sc1 = ax.scatter(x0, y0, s=10, linewidth=0, facecolor='green', label='particles')
        loc = self.roadmap.get_loc(self.target1.state)
        self.sc_target1 = ax.scatter([loc[0]], [loc[1]], s=100, marker='*', facecolor='green', label='target')

        if self.num_targets > 1:
            x0 = self.rbpf.best[1].X[:,0]
            y0 = self.rbpf.best[1].X[:,1]
            self.sc2 = ax.scatter(x0, y0, s=10, linewidth=0, facecolor='blue', label='particles')
            loc = self.roadmap.get_loc(self.target2.state)
            self.sc_target2 = ax.scatter([loc[0]], [loc[1]], s=100, marker='*', facecolor='blue', label='target')
        #
        self.agent.init_plot(ax)
        ax.legend()
        #ax.plot([50, 60], [50, 50], marker='o', ls='None')

        # ax.set_xlim(-5, 250)
        # ax.set_ylim(-5, 120)
        ax.set_aspect('equal')
        # plt.plot([0,1,1,2])
        plt.show()
        self.fig.canvas.draw()
    #
    def update_visualization(self):
        locs1 = self.rbpf.best[0].X[:,:2]
        self.sc1.set_offsets(locs1)
        self.sc_target1.set_offsets(self.target1.loc)
        if self.num_targets > 1:
            locs2 = self.rbpf.best[1].X[:,:2]
            self.sc2.set_offsets(locs2)
            self.sc_target2.set_offsets(self.target2.loc)
        #
        plt.show()
        self.fig.canvas.draw()
    #
    def get_starting_pos(self, agent=True):
        edge_start = random.choice(list(self.roadmap.graph.keys()))
        edge_end = random.choice(list(self.roadmap.graph[edge_start].keys()))
        dist_along_edge = np.random.random()
        if agent:
            starting_pos = np.array(edge_start) + (np.array(edge_end)-np.array(edge_start))*dist_along_edge
            starting_pos = (starting_pos[0], starting_pos[1])
            return starting_pos
        else:
            return (edge_start, edge_end), dist_along_edge
        #
    #
    def reset(self, visualize=False):
        edge_t1, x0_t1 = self.get_starting_pos(False)
        self.target1 = Particle(self.roadmap, v0=10, e0=edge_t1, x0=x0_t1, dt=self.dt, sigma=2)
        if self.num_targets > 1:
            edge_t2, x0_t2 = self.get_starting_pos(False)
            self.target2 = Particle(self.roadmap, v0=10, e0=edge_t2, x0=x0_t2, dt=self.dt, sigma=2)
        #
        pf_args = {'roadmap':self.roadmap, 'num_particles':self.N, 'dt':self.dt,
                   'v0':self.v0, 'sigma':self.sigma, 'P_fa':self.P_fa,
                   'P_miss':self.P_miss}
        #
        self.rbpf = RBPF(self.roadmap, 10, self.num_targets, pf_args)
        agent_starting_pos = self.get_starting_pos()
        self.agent = Agent(agent_starting_pos, 100, 50, 40, 30, self.roadmap, dt=self.dt)
        self.roll_steps = 0

        # X = np.array([self.rbpf.best[ii].X for ii in range(len(self.rbpf.best))])
        X = np.array([ii.X for ii in self.rbpf.best])
        # set_trace()
        self.max_entropy = calc_entropy(self.roadmap.graph, X)
        self.last_entropy = self.max_entropy
        self.min_entropy = self.max_entropy
        self.entropy_list = []
        if visualize:
            # print('test')
            self.setup_visualization()
        #
        return self.get_state()
    #
    def get_agent_greedy_waypoint(self):
        entropies = self.rbpf.calc_entropies()
        return self.agent.get_next_waypoint(entropies, self.rbpf.best)
    #
    def get_state(self):
        # particles = np.array([X.X.flatten() for X in self.rbpf.best]).flatten()

        # TODO pull, elements 0,1,6,7
        get_pos_vel = [0,1,6,7]

        # particles = np.array([X.X[:,get_pos_vel] for X in self.rbpf.best])
        particles = self.rbpf.get_state()
        # flattened_particles = np.array([X.X.flatten() for X in self.rbpf.best]).flatten()
        X = np.array([ii.X for ii in self.rbpf.best])

        return np.concatenate(
            (np.array([
                self.agent.pos[0], self.agent.pos[1], self.agent.inertial_heading
            ]), particles,
            np.array(calc_entropy(self.roadmap.graph, X)).flatten(),
            # ]), flattened_particles,
            np.array(self.nodes).flatten(), np.array(self.edges).flatten()))
        #
    #
    def get_state_dimensions(self):
        return self.get_state().shape[0]
    #
    def get_action_dimensions(self):
        # return self.roadmap.get_action_dimensions() - 1
        return self.roadmap.max_neighbors
    #
    def step(self, action_index, wp=None):
        # seg_steps = 0
        # segment_entropy = 0
        # action = sorted(self.roadmap.graph.keys())[action_index]
        # action = sorted(self.roadmap.graph.keys())
        # # print(action)
        # set_trace()
        # options = self.agent.get_options()
        # # action = self.agent.get_options()[action_index]
        # action = options[action_index]
        # if wp is None:
        #     node_id_current = self.roadmap.nodes_sorted.index(tuple(self.agent.end_node))
        #     node_id_cmd = self.roadmap.select_act[node_id_current, action_index]
        #
        #     bad_choice = np.isnan(node_id_cmd)
        #     # print(bad_choice, action, options)
        #     if bad_choice:
        #         return self.get_state, 0, bad_choice
        #     #
        #
        #     action = self.roadmap.nodes_sorted[node_id_cmd.astype(int)]
        #
        # else:
        #     action = wp
        #     bad_choice = False
        #
        self.agent.setHeading(action_index)
        # print(self.agent.pos)
        # self.agent.set_next_intersection(action)
        # print("waypoints", self.agent.end_node, self.agent.pos, len(options), action)


        # while not self.agent.is_at_intersection():
        entropy = self.single_step()
        self.entropy_list.append(np.linalg.norm(entropy))
        self.roll_steps += 1
        # seg_steps += 1
        entropy = np.linalg.norm(entropy)
            #
        #
        # reward = 0
        # if seg_steps != 0:
        #     avg_segment_entropy = segment_entropy / seg_steps
        #     # print(avg_segment_entropy)
        if entropy > self.max_entropy:
            self.max_entropy = entropy
        # reward = self.max_entropy - entropy
        # if reward
        reward = entropy - self.last_entropy
        if entropy < self.min_entropy:
            self.min_entropy = entropy
            reward += 10
        # print(self.max_entropy, entropy, reward)
        #
        # if reward > 31:
        #     reward = 1.12**reward
        # print( avg_segment_entropy, reward)

        x = self.agent.pos[0]
        y = self.agent.pos[1]
        bad_choice =  (x < -20 or x > 220 or y < -20 or y > 120)

        return self.get_state(), reward, bad_choice
    #
    def get_entropy_over_sim(self):
        return self.entropy_list

    def single_step(self):
        Ts = self.dt
        self.rbpf.predict()
        self.target1.predict()
        dist1 = np.linalg.norm(self.target1.loc - self.agent.pos)
        if self.num_targets > 1:
            self.target2.predict()
            dist2 = np.linalg.norm(self.target2.loc - self.agent.pos)
        else:
            dist2 = self.agent.fov + 1
        #
        if self.roll_steps % int(1/Ts) == 0 and self.roll_steps != 0:
            if (dist1 < self.agent.fov) != (dist2 < self.agent.fov):
                if dist1 < self.agent.fov:
                    z = mvn.rvs(self.target1.loc, self.R)
                    self.rbpf.update(z, self.R, lone_target=True,
                                     radius=self.agent.fov*0.75)
                    #
                #
                if dist2 < self.agent.fov:
                    z = mvn.rvs(self.target2.loc, self.R)
                    self.rbpf.update(z, self.R, lone_target=True, radius=self.agent.fov*0.75)
                #
            elif (dist1 < self.agent.fov) and (dist2 < self.agent.fov):
                z = mvn.rvs(self.target1.loc, self.R)
                self.rbpf.update(z, self.R, lone_target=False)
                z = mvn.rvs(self.target2.loc, self.R)
                self.rbpf.update(z, self.R, lone_target=False)
            else:
                self.rbpf.neg_update(self.agent.pos, radius=self.agent.fov*0.75)
            #
        #
        pfs = self.rbpf.best
        #TODO replace agent stuff
        if self.fig:
            self.update_visualization()
        #
        self.agent.update(pfs)

        # X = np.array([self.rbpf.best[ii].X for ii in range(len(self.rbpf.best))])
        X = np.array([ii.X for ii in self.rbpf.best])
        # H += [calc_entropy(self.roadmap.graph, X)]

        return calc_entropy(self.roadmap.graph, X)
    #
#
def calc_entropy( r, X, res=5):
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
    dists = np.linalg.norm(X[:, :, :2] - X[:, :, 2:4], axis=-1)

    h = 0
    for start in r.keys():
        for end in r[start].keys():
            length = r[start][end]
            bin_start = 0.0
            # find the particles on this road segment
            on_edge = np.all(X[:, :, 2:6] == start + end, axis=-1)
            while bin_start < length:
                in_bin = np.all([dists >= bin_start, dists <= bin_start + res], axis=0)
                count = np.sum(np.all([on_edge, in_bin], axis=0))
                p = count / (N*M)
                if p > 0:
                    h -= p*np.log(p)
                #
                bin_start += res
            #
        #
    #
    return h
#

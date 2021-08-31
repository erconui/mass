from __future__ import division

import random
import copy
import numpy as np
from pf import PF

class RBPF(object):
    def __init__(self, roadmap, num_particles, max_vehicles, pf_args):
        self._roadmap = roadmap
        self._N = num_particles
        self._max_vehicles = max_vehicles
        self.X = [[PF(**pf_args) for jj in range(self._max_vehicles)] for ii in range(self._N)]
        self.best = self.X[0]
        self.no_measurements = True
    #
    def lowVarSample(self, ww):
        Xbar = []
        M = self._N
        r = np.random.uniform(0, 1/M)
        c = ww[0]
        ii = 0
        last_i = ii
        unique = 1
        for m in range(M):
            u = r + m/M
            while u > c:
                ii += 1
                c = c + ww[ii]
            #
            new_x = copy.deepcopy(self.X[ii])
            Xbar.append(new_x)
            if ii == self.best_idx:
                self.best = new_x
            #
            if last_i != ii:
                unique += 1
            #
            last_i = ii
        #
        self.X = Xbar
        return unique
    #
    def predict(self):
        # propagate each bank of particle filters
        [[xi.predict() for xi in xx] for xx in self.X]
    #
    def update(self, z, R, lone_target, radius=None, p_fa=None):
        # print("updating")
        ww = np.zeros(self._N)

        for ii, xx in enumerate(self.X):
            if self.no_measurements:
                tt = 0
            else:
                # get the likelihood that the measurement came from each target
                ll = np.array([xi.get_measurement_likelihood(z, R) for xi in xx])

                # normalize the likelihoods so we can randomly choose a corresponding target
                # with some smart probabilites
                ll = ll/np.sum(ll)
                # t = np.where(np.random.multinomial(1, l) == 1)[0][0]
                tt = np.random.choice(range(len(ll)), p=ll)
                # print(t)
            #
            ww[ii] = xx[tt].get_measurement_likelihood(z, R)
            xx[tt].update(z, R)
            if lone_target:
                for jj, xi in enumerate(xx):
                    if tt != jj:
                        xi.neg_update(z, radius)
                    #
                #
            #
        #
        self.no_measurements = False


        # logsumexp
        max_w = np.max(ww)
        ww = np.exp(ww-max_w)
        # for code simplicity, normalize the weights here
        ww = ww/np.sum(ww)
        # print("best: {}={}".format(np.argmax(w), np.max(w)))

        self.best_idx = np.argmax(ww)
        self.best = self.X[self.best_idx]
        unique = self.lowVarSample(ww)
        # print(unique)
    #
    def neg_update(self, z, radius):
        [[xi.neg_update(z, radius) for xi in xx] for xx in self.X]
    #
    def calc_entropies(self, res=5):
        return [self.calc_entropy(np.array([pf.X])) for pf in self.best]
    #
    def calc_entropy(self, X=None, res=5):
        """Returns the entropy of the estimate in nats

            r -- roadmap graph on which the particles exist
            X -- state of each particle, shape=(M, N, 12),
                 M = number of targets
                 N = number of particles
            """
        if X is None:
            # X = np.array([self.best[ii].X for ii in range(len(self.best))])
            X = np.array([ii.X for ii in self.best])
        #
        ## x, y, start_x, start_y, end_x, end_y, direction_x, direction_y, distance, sigma, w
        M = X.shape[0]
        N = X.shape[1]
        # calculate the distance of each particle from the beginning of its road segment
        dists = np.linalg.norm(X[:, :, :2] - X[:, :, 2:4], axis=-1)

        hh = 0
        for start in self._roadmap.graph.keys():
            for end in self._roadmap.graph[start].keys():
                length = self._roadmap.graph[start][end]
                bin_start = 0.0
                # find the particles on this road segment
                on_edge = np.all(X[:, :, 2:6] == start + end, axis=-1)
                while bin_start < length:
                    in_bin = np.all([dists >= bin_start, dists <= bin_start + res], axis=0)
                    count = np.sum(np.all([on_edge, in_bin], axis=0))
                    pp = count / (N*M)
                    if pp > 0:
                        hh -= pp*np.log(pp)
                    #
                    # if hh == 0:
                    #     print(count, pp)
                    bin_start += res
                #
            #
        #
        return hh
    #

    def get_state(self, res=.1):
        X = np.array([ii.X for ii in self.best])

        M = X.shape[0]
        N = X.shape[1]

        bins = []

        for target in range(X.shape[0]):
            percentages = np.linalg.norm((X[target,:,:2] - X[target,:,2:4]), axis=-1)/X[target,:,8]

            for start in sorted(self._roadmap.graph.keys()):
                for end in sorted(self._roadmap.graph[start].keys()):
                    leng = self._roadmap.graph[start][end]
                    bin_start = 0.0

                    on_edge = np.all(X[target,:,2:6] == start+end, axis=-1)
                    while bin_start < 1:
                        in_bin = np.all([percentages >= bin_start, percentages <= bin_start + res], axis=0)
                        count = np.sum(np.all([on_edge, in_bin], axis=0))
                        bins.append(count)

                        bin_start += res
        return bins

#

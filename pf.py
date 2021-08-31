from __future__ import division

import random
import copy
import numpy as np
from scipy.stats import multivariate_normal as mvn

class PF(object):
    def __init__(self, roadmap, num_particles, dt, v0=10., sigma=4, x0=None, P_fa=0.1, P_miss=0.1):
        self._roadmap = roadmap
        self._N = num_particles
        self._v0 = v0
        # particle shape
        ## x, y, start_x, start_y, end_x, end_y, direction_x, direction_y, distance, sigma, w
        # distance = end - start
        # sigma = noise of particles, how fast they spread out
        # w = weight of particles
        # pos_x, pos_y, vel_x, vel_y,
        self.X = np.ndarray(shape=(self._N, 11))

        if x0 is None:
            #uniformly distribute the particles on the roadway
            for ii in range(self._N):
                a = random.choice(list(self._roadmap.graph.keys()))
                b = np.array(random.choice(list(self._roadmap.graph[a].keys())))
                a = np.array(a)
                vector = b - a
                loc = a + vector * random.random()
                distance = np.linalg.norm(vector)
                velocity = v0 * vector/distance
                self.X[ii] = [loc[0], loc[1], a[0], a[1], b[0],
                                    b[1], velocity[0], velocity[1], distance, sigma, 1/self._N]
                #
            #
        else:
            raise NotImplementedError
        #
        self.best = self.X[0]
        self._dt = dt
        self._P_fa = P_fa
        self._P_miss = P_miss
    #
    def get_particle_distance_from_target(self, loc, acceptable_distance):
        avg = 0
        num_in_distance = 0
        dist = np.linalg.norm(self.X[:,0:2] - loc, axis=1)
        num_in_distance = len(np.where(dist < acceptable_distance)[0])
        avg = np.average(dist)
        return avg, num_in_distance
    #velocity = self.X[:,6:8]
    def get_max_particle_density(self):
        overall_avg = 0
        min_avg = 9999999999
        for ii in range(len(self.X)):
            dist = np.linalg.norm(self.X[:,0:2] - self.X[ii,0:2], axis=1)
            avg_dist = np.average(dist)
            overall_avg += avg_dist
            if avg_dist < min_avg:
                min_avg = avg_dist
            #
        #
        return overall_avg, min_avg
    #
    def get_edge_particle_density(self):
        edges = {}
        edges_present, count = np.unique(self.X[:, 2:6], return_counts=True, axis=0)
        for ii in range(len(edges_present)):
            start_edge = (edges_present[ii][0], edges_present[ii][1])
            end_edge = (edges_present[ii][2], edges_present[ii][3])
            edges[(start_edge, end_edge)] = count[ii]
        #
        return edges
    #
    def get_measurement_likelihood(self, z, R):
        return np.sum(mvn.pdf(self.X[:,:2], z, R))
    #
    def low_var_sample(self):
        M = self._N
        r = np.random.uniform(0,1/M)
        c = self.X[0,10]
        new_particles = np.zeros_like(self.X)
        ii = 0
        last_i = 1
        unique = 1
        insert_index = 0
        for m in range(M):
            u = r + m/M
            while u > c:
                ii += 1
                c = c + self.X[ii,10]
            #
            new_particles[insert_index] = copy.deepcopy(self.X[ii])
            insert_index += 1
            if last_i != ii:
                unique += 1
            #
            last_i = ii
        #
        self.X = new_particles
        return unique
    #
    def predict(self, timestep=1):
        # old_particles = deepcopy(self.X)

        # noise in movement based on sigma (not working)
        n = np.random.normal(scale=self.X[:,9:10])
        loc = self.X[:,0:2]
        velocity = self.X[:,6:8]

        n = np.divide(n,np.linalg.norm(velocity,axis=1).reshape(-1,1))

        loc[...] += velocity * (1+n) * timestep * self._dt
        update_vector = np.linalg.norm(self.X[:,0:2] - self.X[:,2:4], axis=1) \
          > self.X[:,8]
        #

        for ii in np.where(update_vector)[0]:
            a = (self.X[ii,2], self.X[ii,3])
            b = (self.X[ii,4], self.X[ii,5])

            dest_list = list(self._roadmap.graph[b].keys())
            dest_list.remove(a)
            a = self.X[ii,4:6]
            b = np.array(random.choice(dest_list))
            dir_vector = b - a
            distance = np.linalg.norm(dir_vector)
            velocity = self._v0 * dir_vector/distance
            self.X[ii,0:9] = [a[0], a[1], a[0], a[1], b[0],
                         b[1], velocity[0], velocity[1], distance]
            #
        #
    def update(self, z, R, p_fa=None):
        weight_addon = self._P_fa/self._roadmap.total_length
        w = (1. - self._P_fa)*mvn.pdf(self.X[:,0:2], z, R) + weight_addon


        w = np.log(w)

        max_w = np.max(w)
        w = np.exp(w-max_w)
        # for code simplicity, normalize the weights here
        w = w/np.sum(w)

        self.best_idx = np.argmax(w)
        self.best = self.X[self.best_idx]
        self.X[:,10] = w

        unique = self.low_var_sample()
    #
    def neg_update(self, z, radius):
        self.X[:,10] = 1
        update_vector = np.linalg.norm(self.X[:,0:2] - z, axis=1) < radius
        self.X[update_vector,10] = self._P_miss
        self.X[:,10] /= np.sum(self.X[:,10])
        unique = self.low_var_sample()
    #
#
class PFJupyter(object):
    def __init__(self, roadmap, num_particles, dt, v0=10., sigma=4, x0=None, P_fa=0.1, P_miss=0.05, one_edge=False):
        self._roadmap = roadmap
        self.one_edge = one_edge
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
            dist = np.linalg.norm(self.X[i,0:2] - self.X[i,3:5]) - self.X[i,9]
            a = (self.X[i,3], self.X[i,4])
            b = (self.X[i,5], self.X[i,6])

            dest_list = list(self._roadmap.graph[b].keys())
            if not self.one_edge:
                dest_list.remove(a)
            a = self.X[i,5:7]
            b = np.array(random.choice(dest_list))
            vector = b - a
            distance = np.linalg.norm(vector)
            vector = vector/distance
            loc = (a[0] + vector[0]*dist, a[1] + vector[1]*dist)
            self.X[i,0:10] = [loc[0], loc[1], self._v0, a[0], a[1], b[0],
                           b[1], vector[0], vector[1], distance]

    def update(self, z, R, p_fa=None):
        weight_addon = self._P_fa/self._roadmap.total_length
        w = (1. - self._P_fa)*mvn.pdf(self.X[:,0:2], z, R) + weight_addon

        # print(np.min(w))
        w = np.log(w)

        max_w = np.max(w)
        w = np.exp(w-max_w)
        # for code simplicity, normalize the weights here
        w = w/np.sum(w)

        self.best_idx = np.argmax(w)
        self.best = self.X[self.best_idx]
        self.X[:,11] = w

        unique = self.low_var_sample()

    def neg_update(self, z, radius):
        self.X[:,11] = 1
        update_vector = np.linalg.norm(self.X[:,0:2] - z, axis=1) < radius
        self.X[update_vector,11] = self._P_miss
        self.X[:,11] /= np.sum(self.X[:,11])
        unique = self.low_var_sample()

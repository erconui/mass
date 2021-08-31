import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
from roadmap import Roadmap
import random

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

class Modes(object):
    def __init__(self, roadmap, v0, dt, e0=None, x0=None, sigma=4, merging_allowed=False):
        self._roadmap = roadmap
        self._merging_allowed = merging_allowed
        if e0 is None:
            options = []
            for a in self._roadmap.graph.keys():
                for b in self._roadmap.graph[a].keys():
                    if (a,b) not in options:
                        options.append((a,b))
#             a = random.choice(list(self._roadmap.graph.keys()))
#             b = random.choice(list(self._roadmap.graph[a].keys()))
#             e = (a, b)
            e = random.choice(options)
        else:
            e = e0
        if x0 is None:
            x = random.random()
        else:
            x = x0
        self.modes = np.array([[
            e[0][0], e[0][1], e[1][0], e[1][1], self._roadmap.graph[e[0]][e[1]], x, 1.
        ]])
        self._v = v0
        self._sigma = sigma
        self._dt = dt

    def predict(self):
        n = 0#np.random.normal(scale=self._sigma, size=self.modes.shape[0])#.reshape(self.modes.shape[0],1)
        self.modes[:,5] += ((self._v + n))*self._dt/self.modes[:,4]
        split = np.where(self.modes[:,5]>1)[0]
        for i in reversed(split):
            start = (self.modes[i,2], self.modes[i,3])
            x = self.modes[i,5] - 1
            destinations = list(self._roadmap.graph[start])
            destinations.remove((self.modes[i,0],self.modes[i,1]))
            val = self.modes[i,6] / len(destinations)
            for dest in destinations:
                self.modes = np.append(self.modes, [[start[0], start[1], dest[0], dest[1], self._roadmap.graph[start][dest], x, self.modes[i,6] / len(destinations)]], axis=0)
            self.modes = np.delete(self.modes, i, axis=0)
        if self._merging_allowed:
            edges_occupied, counts = np.unique(self.modes[:,0:4], return_counts=True, axis=0)
            for idx in range(len(counts)):
                if counts[idx] > 1:
                    duplicates = (self.modes[:,0:4]==edges_occupied[idx]).all(axis=1).nonzero()[0]
                    self.modes[duplicates[0],6] = self.modes[duplicates,6].sum()
                    self.modes = np.delete(self.modes, duplicates[1:],axis=0)


def modesChanged(last_count, count):
    tmpCount = [c for c in count]
    for val in last_count:
        if val not in tmpCount:
            return True
        tmpCount.remove(val)
    if len(tmpCount) > 0:
        return True
    return False

def updateModeData(modes_over_time, count):
    length = np.mean([len(modes_over_time[key]) for key in modes_over_time])
    if np.isnan(length):
        length = 0
    length = int(length+1)
    for key in modes_over_time:
        modes_over_time[key].append(0)
    for val in count:
        if val[0] not in modes_over_time:
            modes_over_time[val[0]] = [0 for i in range(length)]
        modes_over_time[val[0]][-1] = val[1]
def getModeDataOverTime(dt, duration, dist_e, layout, merging_allowed=False):
#     dist_e = 100
    nodes, edges = createGridLayout(layout[0],layout[1],dist_e, dist_e)
    r = Roadmap(nodes, edges, rotate=False)
#     dt = .1
    modes = Modes(r, 10, dt, merging_allowed=merging_allowed)
    modes_index = 0
#     duration = 600
    last_count = []
    modes_over_time = {}
    for i in range(int(duration/dt)):
        modes.predict()
        seen = []
        count = []
        for j in range(modes.modes.shape[0]):
            val = modes.modes[j,6]
            if val not in seen:
                seen.append(val)
                count.append((val, np.where(modes.modes[:,6]==val)[0].shape[0]))
        updateModeData(modes_over_time, count)
    return modes_over_time
def getAvgModeDataOverTime(dt, duration, dist_e, layout, num_runs, merging_allowed=False):
    mode_data = {}
    for idx in tqdm(range(num_runs)):
        modes_over_time = getModeDataOverTime(dt, duration, dist_e, layout, merging_allowed=merging_allowed)
        for key in modes_over_time.keys():
            if key not in mode_data:
                mode_data[key] = []
            mode_data[key].append(modes_over_time[key])
    for key in mode_data.keys():
        mode_data[key] = np.mean(mode_data[key], axis=0)
    return mode_data

def plotModeData(ax, modes_over_time, dt, duration, label=''):
    last_index = int(duration/dt)
    x = [i*dt for i in range(last_index)]
    for key,data in modes_over_time.items():
        i = next((i for i, x in enumerate(data) if x > 0), -1)
        if i >= 0:
            ax.plot(x[i:last_index], data[i:last_index], label='{} {}'.format(label,key))
    # ax.legend()
    ax.set_xlabel("Time")
    ax.set_ylabel("Probability")
    ax.set_title('{} map'.format(label))
    # plt.show()

def get_time(t, ts, offset, initial_range):
    time = t - (ts + offset)
    time[np.where(time<0)] = 0
    time[np.where(time>initial_range)] = initial_range
    return time

def getModeBirthDeathProb(t, ts, offsets, initial_range):
    probs = get_time(t, ts, 0, initial_range)
    for idx in range(len(offsets)):
        probs -= (1/len(offsets))*get_time(t, ts, offsets[idx], initial_range)
    probs /= initial_range
    return probs

def get_layer(key, initial_prob, initial_range, prob, ts, map_dict, t):
    test_data = {}
    prob_data = {}
    probs = map_dict[key]['prob']
    base_offsets = map_dict[key]['offsets']
    options = map_dict[key]['options']
    sum_probs = 0
    for idx in range(len(options)):
        next_key = prob * len(map_dict[options[idx]]['options'])
        if next_key not in test_data:
            test_data[next_key] = []
            prob_data[next_key] = np.zeros(len(t))
        if key is 'base':
            initial_prob = map_dict[options[idx]]['prob']
            initial_range = map_dict[options[idx]]['range']
        test_data[next_key].append([initial_prob,initial_range,ts+base_offsets[idx],options[idx]])

        offsets = map_dict[options[idx]]['offsets']
        sum_probs += initial_prob
        prob_data[next_key] += initial_prob*(1/prob)*getModeBirthDeathProb(t,ts+base_offsets[idx],offsets,initial_range)
    return test_data, prob_data

def get_recursive_layers(keys, data, prob_data, t, map_dict):
    next_keys_list = []
    new_modes = False
    for key in keys:
        for idx in data[key]:
            layer_key = idx[3]
            initial_prob = idx[0]
            initial_range = idx[1]
            prob = key
            ts = idx[2]
            if ts < t[-1]:
                new_modes = True
                tmp_data, tmp_prob_data = get_layer(layer_key, initial_prob, initial_range, prob, ts, map_dict, t)
                for tmp_key in tmp_data.keys():
                    if tmp_key not in data:
                        data[tmp_key] = []
                        prob_data[tmp_key] = np.zeros(len(t))
                        next_keys_list.append(tmp_key)
                    data[tmp_key].extend(tmp_data[tmp_key])
                    prob_data[tmp_key] += tmp_prob_data[tmp_key]
    if new_modes:
        get_recursive_layers(next_keys_list, data, prob_data, t, map_dict)

def getModeProbabilities(map_dict, dt, max_duration):
    t = np.array([idx*dt for idx in range(int(max_duration/dt))])
    test_data, prob_data = get_layer('base',0,0,1,0,map_dict,t)
    get_recursive_layers(list(test_data.keys()), test_data, prob_data, t, map_dict)
    prob_data_1 = np.ones(len(t))
    keys = list(prob_data)
    if 1 in keys:
        keys.remove(1)
    for key in keys:
        prob_data_1 -= prob_data[key]
        # print(key,prob_data_1[50])
    prob_data[1] = prob_data_1
    # for idx in range(len(map_dict['base']['prob'])):
        # ranges = map_dict['base']['range']
        # base_probs = map_dict['base']['prob']
        # offsets = map_dict['base']['offsets']
        # options = map_dict['base']['options']
        # prob_data[1] -= base_probs[idx]*(1./ranges[idx])*get_time(t,0,offsets[idx],ranges[idx])
    return t, test_data, prob_data

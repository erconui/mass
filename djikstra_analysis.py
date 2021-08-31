import numpy as np
from tqdm import tqdm
# from tqdm import tqdm_notebook as tqdm
from entropy_roadmap import Roadmap
from entropy_particle import Particle

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

def getAvgDistance(r):
    vals = []
    for node in sorted(r.graph.keys()):
        avg_dist = np.mean(DjikstraGraph(r.graph, node))
        vals.append(avg_dist)
    return np.mean(vals)

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
    t1 = targets[0]
    t2 = targets[1]
    # target3 = targets[2]
    # target4 = targets[3]
    # dist = getShortestPath(graph, t1._e, t1._x, t2._e, t2._x, max_depth)

    min_dist = np.inf
    for t1 in targets:
        unvisited = [target for target in targets]
        unvisited.remove(t1)
        sequences = getSequences(graph, t1, unvisited, max_depth)
        # print(sequences)
        for sequence in sequences:
            return_dist = getShortestPath(graph, t1._e, t1._x, sequence[1][-1]._e, sequence[1][-1]._x, max_depth)
            sequence[0] += return_dist
            sequence[1].insert(0,t1)
            sequence[2].append(return_dist)
            # print(sequence)
            min_dist = min(min_dist, sequence[0])

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

    return min_dist/len(targets)

def getAverageDistanceOverSim(v0, dt, r, duration=40, num_targets=2, max_depth=7):
#     target1 = Particle(r, v0=v0, dt=dt, sigma=4)
#     target2 = Particle(r, v0=v0, dt=dt, sigma=4)
    targets = []
    for i in range(num_targets):
        targets.append(Particle(r, v0=v0, dt=dt, sigma=4, name=i))
    total_distance = np.zeros(num_targets - 1)
#     print(targets)
    for i in range(duration):
#         print('test', i)
#         shortest_path_value = getShortestPath(r.graph, target1._e, target1._x, r, target2._e, target2._x, max_depth)
        for i in range(1, num_targets):
#             print(1-i)
            shortest_path_value = getShortestRoundTrip(r.graph, targets[:i+1], max_depth)
            total_distance[i-1] += shortest_path_value/(i+1)
#         print('dist',total_distance)
#         target1.predict()
#         target2.predict()
        for target in targets:
            target.predict()
    return total_distance/duration

def getAverageErrorForLayout(layout, edge_length, v0, dt, num_runs, sim_duration, num_targets):
    nodes, edges = createGridLayout(layout[0], layout[1], edge_length, edge_length)
    r = Roadmap(nodes, edges)
    total_distance = np.zeros(num_targets - 1)
    for j in range(num_runs):
#         print(j)
        avg_distance = getAverageDistanceOverSim(v0, dt, r, duration=sim_duration, num_targets=num_targets)
        total_distance += avg_distance

    estimated_dist = getAvgDistance(r)
#     print(estimated_dist, total_distance/num_runs)
#     print(estimated_dist, total_distance/num_runs, num_runs)
    error = estimated_dist*np.ones(num_targets - 1) - total_distance/num_runs
    return error

def getAverageErrorForLayoutType(layout, edge_length, v0, dt, num_iterations, num_runs, sim_duration, num_targets):
    dt = .1
    total_error = np.zeros(num_targets - 1)
    for i in tqdm(range(num_iterations)):
        error = getAverageErrorForLayout(layout, edge_length, v0, dt, num_runs, sim_duration, num_targets=num_targets)
#         print(error)
        total_error += error
#     print(total_error.shape)
    return total_error/num_iterations

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
            nodes.append((
                x_val + np.random.uniform(low=min_edge_length, high=max_edge_length),
                y_val + np.random.uniform(low=min_edge_length, high=max_edge_length)))

    for i in range(y):
        for j in range(x-1):
            edges.append((j+x*i,j+1+x*i))

    for i in range(y-1):
        for j in range(x):
            edges.append((j+x*i,j+x*(i+1)))
    return [nodes, edges]

def getAverageDistanceOverSim(v0, dt, r, max_depth, num_targets, duration=40):
    target1 = Particle(r, v0=v0, dt=dt, sigma=4)
    target2 = Particle(r, v0=v0, dt=dt, sigma=4)
    targets = [Particle(r, v0=v0, dt=dt, sigma=4) for i in range(num_targets)]
    total_distance = np.zeros(num_targets - 1)
    for i in range(duration):
#         print('test', i)
        for i in range(num_targets - 1):
            # print(i, num_targets-1)
            shortest_path_value = getShortestRoundTrip(r.graph, targets[:i+2], max_depth)
            total_distance[i] += shortest_path_value
        # print(total_distance)
        for target in targets:
            target.predict()
        # target1.predict()
        # target2.predict()
    return total_distance/duration

def getAverageErrorForLayout(layout, edge_length, v0, dt, num_runs, num_targets, sim_duration, max_depth):
    nodes, edges = createGridLayout(layout[0], layout[1], 0, edge_length)
    r = Roadmap(nodes, edges)
    total_distance = np.zeros(num_targets - 1)
    for j in tqdm(range(num_runs)):
        avg_distance = getAverageDistanceOverSim(v0, dt, r, max_depth, num_targets, duration=sim_duration)
        total_distance += avg_distance

    estimated_dist = getAvgDistance(r)
#     print(estimated_dist, total_distance/num_runs, num_runs)
    error = estimated_dist*np.ones(num_targets - 1) - total_distance/num_runs
    return error

def getAverageErrorForLayoutType(layout, edge_length, v0, dt, num_iterations, num_runs, sim_duration):
    dt = .1
    total_error = 0
    for i in tqdm(range(num_iterations)):
        error = getAverageErrorForLayout(layout, edge_length, v0, dt, num_runs, sim_duration)
#         print(error)
        total_error += error
    return total_error/num_iterations

## Get the difference between the djikstra estimate and the average distance between targets
# over an extended simulation time
## Get average distance between any two particles on given map
Va = 30
data = []
for i in reversed(range(1,5)):
    print()
    dist_e = i*100
    layouts = [
        # (2,2),#(2,3),(2,4),(2,5),(2,6),(2,7),(2,8),(2,9),
#         (3,3),#(3,4),(3,5),(3,6),(3,7),(3,8),(3,9),
        (4,4),(4,5),(4,6),(4,7),(4,8),(4,9),
        # (5,5),#(5,6),(5,7),(5,8),(5,9),
#         (6,6),#(6,7),(6,8),(6,9),
#         (7,7),#(7,8),(7,9),
#         (8,8),#(8,9),
        # (9,9)
    ]

    for layout in layouts:
        dt = .1

#         total_value = 0
        num_iterations=1
        num_runs = 100
        sim_time = 1000
        num_targets = 6
        total_error = np.zeros(num_targets - 1)
        max_depth = 10
        for i in tqdm(range(num_iterations)):
            error = getAverageErrorForLayout(layout, dist_e, 10, dt, num_runs, num_targets, sim_time, max_depth)
            total_error += error
#             total_value += shortest_path_value
#         total_value += best_value
        data.append([
            dist_e,
            layout,
#             total_value/num_runs,
#             getAvgDistance(r),
#             abs(total_value/num_runs - getAvgDistance(r)),
            total_error /num_runs
#             dist_e*len(edges)
        ])
        print("EdgeLength: {} Layout: {} error: {}".format(
            data[-1][0], data[-1][1], data[-1][2]
        ))

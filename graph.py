import numpy as np
from itertools import product


def print_graph(g):
    for (n,c) in g.items():
        print(n, '\t| ', len(c), ' |', c)


def find_path(graph, start, end, path=[]):
        path = path + [start]
        if start == end:
            return path
        for node in graph[start]:
            if node not in path:
                newpath = find_path(graph, node, end, path)
                if newpath: return newpath
        return None


def check_pathing(graph, nodes):
    paths = np.eye(len(nodes))
    for i, j in product(nodes, nodes):
        if not i == j:
            if find_path(graph, i, j) is not None:
                paths[i,j] = 1

    return np.mean(paths)


def make_new_graph(num_rooms, num_doors):
    if num_rooms < num_doors + 1:
        raise Exception('Must be more nodes than required connections per node.')

    room_ids = np.arange(num_rooms)

    connections = 1
    pathed = 0
    while np.sum(connections) > 0 or pathed < 1:
        connections = np.full(num_rooms, num_doors)
        graph = dict((n,[]) for n in room_ids)

        for t in range(50*num_rooms):
            i, j = np.random.choice(room_ids, 2, replace=False)
            connected = j in graph[i] or i in graph[j]
            space     = connections[i] > 0 and connections[j] > 0
            symmetric = connections[i] == connections[j]
            if not connected and space and symmetric:
                graph[i] += [j]
                graph[j] += [i]
                connections[i] -= 1
                connections[j] -= 1

        pathed = check_pathing(graph, room_ids)

    return room_ids, graph

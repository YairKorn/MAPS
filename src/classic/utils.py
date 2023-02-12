import numpy as np
from collections import defaultdict

VALID_ACTIONS = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])

# Calculate the best path and entrace point to a specific area
# Returns the path in a list format
def target_Dijkstra(grid, source, cells):
    frontier = defaultdict(lambda: np.iinfo(np.int64).max)
    results = defaultdict(lambda: np.iinfo(np.int64).max)
    frontier[tuple(source)] = 0
    

    while frontier:
        min_cost, next_vertex = np.iinfo(np.int64).max, None
        for vertex in frontier:
            if frontier[vertex] < min_cost:
                min_cost = frontier[vertex]
                next_vertex = vertex
        
        # Reached to the desired area - calculate the path
        if next_vertex in cells:
            path = [next_vertex,]
            while (path[-1] != source).any():
                prevs = np.array([results[tuple(prev)] for prev in path[-1] + VALID_ACTIONS])
                path.append(tuple(path[-1] + VALID_ACTIONS[np.argmin(prevs)]))
            
            return path[:-1][::-1]

        # Else, continue the Dijkstra
        results[tuple(next_vertex)] = min_cost

        childs = next_vertex + VALID_ACTIONS
        for child in childs:
            k, l = child
            if (k >= 0 and k < grid.shape[0]) and (l >= 0 and l < grid.shape[1]) and (grid[k, l] > 0):
                cost = min_cost + grid[k, l]
                if tuple(child) not in results:
                    frontier[tuple(child)] = min(cost, frontier[tuple(child)])
        frontier.pop(tuple(next_vertex))

    return []

# Calculate best path to every cell in the grid
def grid_Dijkstra(grid, source):
    frontier = defaultdict(lambda: np.iinfo(np.int64).max)
    frontier[tuple(source)] = 0
    results  = {tuple(source): 0}

    while frontier:
        min_cost, next_vertex = np.iinfo(np.int64).max, None
        for vertex in frontier:
            if frontier[tuple(vertex)] < min_cost:
                min_cost = frontier[tuple(vertex)]
                next_vertex = vertex
        
        # Reached to the desired area
        results[tuple(next_vertex)] = min_cost

        childs = next_vertex + VALID_ACTIONS
        for child in childs:
            k, l = child
            if (k >= 0 and k < grid.shape[0]) and (l >= 0 and l < grid.shape[1]) and (grid[k, l] > 0):
                cost = min_cost + grid[k, l]
                if tuple(child) not in results:
                    frontier[tuple(child)] = min(cost, frontier[tuple(child)])
        frontier.pop(tuple(next_vertex))

    return results
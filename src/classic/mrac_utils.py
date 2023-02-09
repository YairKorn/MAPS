import numpy as np
import os, yaml
from utils import *


class Area():
    def __init__(self, map: np.ndarray, i: int, j: int, threat_level: int) -> None:
        self.threat_level = threat_level    # Should be a list when it comes to the heterogeneous case
        self.cells = set(self.build_area(map, i, j, threat_level))
        self.assigned = []

    def build_area(self, map: np.ndarray, i: int, j: int, threat_level: int):
        cells = [(i, j)]
        map[i, j] = (map[i, j] + 0.1)

        q = cells.copy()
        while q:
            node = q.pop()
            childs = node + VALID_ACTIONS
            for child in childs:
                k, l = child
                if (k >= 0 and k < map.shape[0]) and (l >= 0 and l < map.shape[1]) and (map[tuple(child)] == threat_level):
                    q.append(child)
                    cells.append((k, l))
                    map[tuple(child)] = (map[tuple(child)] + 0.1)
        
        return cells

    def split_area_between_robots(self):
        # Case that no need to split
        if len(self.assigned) <= 1: 
            return None

        # Case that split is required
        pass #! CONTINUE HERE

class Robot():
    def __init__(self, map: np.ndarray, location) -> None:
        self.area = None        # Assigned area
        self.status = True      # Active (True) or Disabled (False)
        self.location = location

        # Aid data structures
        self.induced_grid = self.create_induced_grid(map)

    def create_induced_grid(self, map: np.ndarray):
        return (map == 0) * (1 / map.size) + map / np.where(map > 0, map, np.inf).min()

    # def select_next_area(self, avail_areas):
    #     pass

    def find_best_path_to_areas(self, areas: list[Area]):
        Dijkstra_results = grid_Dijkstra(self.induced_grid, self.location)
        areas_cost = np.ones((len(areas)))  * np.iinfo(np.int64).max

        for i in range(self.induced_grid.shape[0]):
            for j in range(self.induced_grid.shape[1]):
                for k, area in enumerate(areas): # Consider create a grid of cells belongings
                    if (i, j) in area.cells:
                        areas_cost[k] = min(areas_cost[i], Dijkstra_results[(i, j)])
                        break

        return areas_cost
        
        # return area_Dijkstra(self.induced_grid, self.location, area.cells)


def config2map(config_file, default_path = True):
    path = os.path.join(os.getcwd(), "maps", "coverage_maps") if default_path else ""
    data = yaml.safe_load(open(os.path.join(path, config_file + ".yaml"), 'r'))
    
    # Initialization of the map
    grid = np.zeros(data["world_shape"])
    obs = np.array(data["obstacles_location"])
    grid[obs[:, 0], obs[:, 1]] = -1.0
    ths = np.array(data["threat_location"])
    grid[ths[:, 0].astype(int), ths[:, 1].astype(int)] = ths[:, 2]

    # Initialization of the robots
    robots = [Robot(grid, np.array(loc)) for loc in data["agents_placement"]]
    
    return grid, robots
import numpy as np
import os, yaml, pymetis
from utils import *


class Area():
    def __init__(self, map: np.ndarray, i: int = 0, j: int = 0, threat_level: int = 0, cells: list[tuple] = None) -> None:
        self.threat_level = threat_level    # Should be a list when it comes to the heterogeneous case
        if cells is None:
            self.cells = {cell:0 for cell in self.build_area(map, i, j, threat_level)}
        else:
            self.cells = {cell:0 for cell in cells}
            
        self.assigned = []
        self.build_rep_graph(map)

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

    def build_rep_graph(self, map: np.ndarray):
        VALID_ACTIONS_GRAPH = np.array([1, map.shape[0], -1, -map.shape[0]])

        vertices = sorted([j + i*map.shape[0] for i, j in self.cells])
        rev_vertices = {vertices[k]:k for k in range(len(vertices))}
        verset = set(vertices)

        self.rep_graph = []
        for v in vertices:
            self.rep_graph.append(np.array([rev_vertices[k] for k in VALID_ACTIONS_GRAPH + v if k in verset]))

        # Useful metrices
        self.vertices = vertices
        self.map_size = map.shape

        # for cell in division.keys():
        #     new_map[cell] = division[cell] + 1

    def split_area_between_robots(self, map: np.ndarray):
        # Case that no need to split
        if len(self.assigned) <= 1: 
            return self

        # Case that split is required
        _, split = pymetis.part_graph(len(self.assigned), adjacency=self.rep_graph)
        cells = [[] for _ in range(len(self.assigned))]
        for i in range(len(split)):
            cells[split[i]].append((self.vertices[i]//self.map_size[0],(self.vertices[i]%self.map_size[0])))

        return [Area(map=map, cells=cells[i]) for i in range(len(self.assigned))]

    def mark_as_covered(self, cell):
        assert cell in self.cells
        self.cells





class Robot():
    STATUS = {"IDLE":0, "TRAVELING":1, "COVERING":2, "DISABLED":3}

    def __init__(self, map: np.ndarray, location) -> None:
        self.area = None                    # Assigned area
        self.status = self.STATUS["IDLE"]   # Robot status
        self.location = location            # Current location

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
                        areas_cost[k] = min(areas_cost[k], Dijkstra_results[(i, j)])
                        break

        return areas_cost
        
        # return area_Dijkstra(self.induced_grid, self.location, area.cells)

    def calc_path_to_area(self):
        self.path = target_Dijkstra(self.induced_grid, self.location, self.area.cells) if self.area else []

    def calc_path_in_area(self):
        # A repeated target Dijkstra
        # Consider change area.cells to be dict() rather than set() and mark if a cell is covered
            # OR mark cells on a map and share it between agents
        pass

    def step(self, map):
        if self.status == self.STATUS["TRAVELING"]:
            if self.path:
                self.location = self.path.pop(0)
                
        elif self.status == self.STATUS["COVERING"]:
            if self.path:
                self.location = self.path.pop(0)
                self.area.mark_as_covered(self.location)

        #! #################################
        #!      NEED TO CREATE A MAP OF AREAS
        #!      EACH CELL IS LINKED TO ITS AREA
        #! #################################

        elif self.status == self.STATUS["IDLE"]:
            pass

        else: # Status is "DISABLED"
            pass


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
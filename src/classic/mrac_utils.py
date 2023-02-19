import numpy as np
import os, yaml, pymetis
from utils import *


class Area():
    def __init__(self, map: np.ndarray, i: int = 0, j: int = 0, threat_level: int = 0, cells: list[tuple] = None) -> None:
        self.threat_level = threat_level    # Should be a list when it comes to the heterogeneous case
        if cells is None:
            self.cells = {cell:0 for cell in self.build_area(map, i, j, threat_level)}
        else:
            self.cells = cells
        self.covered_cells = sum(self.cells.values())

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
            e = []
            for k in VALID_ACTIONS_GRAPH + v:
                if (k in verset) and (abs(v//map.shape[0] - k//map.shape[0]) + abs(v%map.shape[0] - k%map.shape[0]) == 1):
                    e.append(rev_vertices[k])
            self.rep_graph.append(np.array(e))

        # Useful metrices
        self.vertices = vertices
        self.map_size = map.shape


    def split_area_between_robots(self, map: np.ndarray, areas_map):
        # Case that no need to split
        if len(self.assigned) <= 1: 
            return [self]

        # Case that split is required
        _, split = pymetis.part_graph(len(self.assigned), adjacency=self.rep_graph)
        cells = [{} for _ in range(len(self.assigned))]
        for i in range(len(split)):
            k, l = self.vertices[i]//self.map_size[0], self.vertices[i]%self.map_size[0]
            cells[split[i]][(k, l)] = self.cells[(k, l)]

        cells = list(filter(None, cells))
        subareas = [Area(map=map, threat_level=self.threat_level, cells=cells[i]) for i in range(len(cells))]
        for i in range(len(cells)):
            for cell in cells[i]:
                areas_map[cell] = subareas[i]

        return subareas


    def mark_as_covered(self, cell):
        assert cell in self.cells
        self.covered_cells += (1 - self.cells[cell])
        self.cells[cell] = 1
        return self.covered_cells == len(self.cells)


    def reallocate_area(self, areas_map: np.ndarray):
        subareas = []
        tmp_map = np.zeros(self.map_size)
        for cell in self.cells:
            tmp_map[cell] = (1 - self.cells[cell])
        
        for cell in self.cells:
            if tmp_map[cell] == 1:
                cells_new = {cell:0}
                tmp_map[cell] = 0

                q = list(cells_new.keys())
                while q:
                    node = q.pop()
                    childs = node + VALID_ACTIONS
                    for child in childs:
                        k, l = child
                        if (k >= 0 and k < self.map_size[0]) and (l >= 0 and l < self.map_size[1]) and (tmp_map[tuple(child)] == 1):
                            q.append(child)
                            cells_new[(k, l)] = 0
                            tmp_map[tuple(child)] = 0
                subareas.append(Area(map=tmp_map, threat_level=self.threat_level, cells=cells_new))

        # Mark subareas on the map
        for i in range(len(subareas)):
            for cell in subareas[i].cells.keys():
                areas_map[cell] = subareas[i]
        return subareas



class Robot():
    STATUS = {"IDLE":0, "TRAVELING":1, "COVERING":2, "DISABLED":3}

    def __init__(self, map: np.ndarray, id: int, alpha: float, location, graph_function: str) -> None:
        self.id = id
        self.area = None                    # Assigned area
        self.status = self.STATUS["IDLE"]   # Robot status
        self.location = location            # Current location
        self.alpha = alpha

        # Aid data structures
        self.graph_function = graph_function
        self.create_induced_grid(map)
        self.past_path = [] #$ DEBUG

    def create_induced_grid(self, map: np.ndarray, cover_map: np.ndarray=None):
        if self.graph_function == "ORIGIN":
            self.induced_grid = (map == 0) * (1 / map.size) + map / np.where(map > 0, map, np.inf).min()
        elif self.graph_function == "SURV":
            threat_cost = -1*np.log(1 - map)
            self.induced_grid = threat_cost + np.where(threat_cost > 1e-3, map, np.inf).min() / map.size
        elif self.graph_function == "OPTIMIZED":
            remaining_cells = map.size - ((cover_map > 0).sum() if cover_map is not None else (map == -1).sum())
            self.induced_grid = map * remaining_cells + self.alpha

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
        

    def calc_path_to_area(self, area=None):
        if area is None:
            self.path = target_Dijkstra(self.induced_grid, np.array(self.location), self.area.cells) if self.area else []
        else:
            self.path = target_Dijkstra(self.induced_grid, np.array(self.location), area.cells)

    def calc_path_in_area(self):
        cells = self.area.cells.copy()
        for cell in self.area.cells:
            if cells[cell]:
                cells.pop(cell)

        while cells:
            self.path += target_Dijkstra(self.induced_grid, np.array(self.path[-1] if self.path else self.location), cells)
            cells.pop(self.path[-1])
        # A repeated target Dijkstra
        # Consider change area.cells to be dict() rather than set() and mark if a cell is covered
            # OR mark cells on a map and share it between agents
        


    def make_a_move(self, map: np.ndarray, areas_map: np.ndarray):
            self.past_path.append(self.location)
            self.location = self.path.pop(0)
            area_status = areas_map[tuple(self.location)].mark_as_covered(tuple(self.location))

            if np.random.rand() < map[tuple(self.location)]:
                print(f"A robot was hit, id: {self.id}")
                self.status = self.STATUS["DISABLED"]   # Set robot as disabled
                if self.area.assigned:
                    self.area.assigned.remove(self)     # Remove the robot from covering the current area
            return area_status

    def step(self, map: np.ndarray, areas_map: np.ndarray):
        area_status = False # "True" if the area is fully covered

        if self.status == self.STATUS["TRAVELING"]:
            area_status = self.make_a_move(map, areas_map)
            
            if not self.path and (self.status != self.STATUS["DISABLED"]):
                self.calc_path_in_area()
                self.status = self.STATUS["COVERING"]

        elif self.status == self.STATUS["COVERING"]:
            area_status = self.make_a_move(map, areas_map)


        elif self.status == self.STATUS["IDLE"]:
            pass

        else: # Status is "DISABLED"; Do nothing
            pass

        if (not self.path) and (self.status != self.STATUS["DISABLED"]):
            self.status = self.STATUS["IDLE"]
        return self.status, area_status


def config2map(config, default_path = True):
    path = os.path.join(os.getcwd(), "maps", "coverage_maps") if default_path else ""
    data = yaml.safe_load(open(os.path.join(path, config.map + ".yaml"), 'r'))
    
    # Initialization of the map
    grid = np.zeros(data["world_shape"])
    obs = np.array(data["obstacles_location"])
    grid[obs[:, 0], obs[:, 1]] = -1.0
    ths = np.array(data["threat_location"])
    grid[ths[:, 0].astype(int), ths[:, 1].astype(int)] = ths[:, 2]

    # Initialization of the robots
    robots = [Robot(grid, id, location=np.array(loc), alpha=config.optim_alpha, graph_function=config.graph_function) for id, loc in enumerate(data["agents_placement"])]
    
    return grid, robots
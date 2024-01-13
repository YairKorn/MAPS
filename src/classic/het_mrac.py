import yaml
import numpy as np
from het_mrac_utils import *
from utils import print_results
from argparse import Namespace
from scipy.optimize import linear_sum_assignment
from argparse import ArgumentParser

#! DOESN'T FIT TO HETEROGENOUS CASE!@!
#!      The area creator keep only one threat level

"""
$ General notes about the algorithm:
1. Non-optimality of the first assignment because it follows an arbitrary order of the robots
2. Coverage path is created before starting to cover the area - better: calc each step, which reduce multiple visits
3. ... in the same manner, robot may go to an area that already was covered
! 4. Consider to adopt the algorithm with the new (better) criteria by changing the heuristic of the induced graph
! 5. Create (3) auto-run script
"""


def areas_creation(map, threats, threat_matrix, num_levels):

    threat_levels = np.ceil((threats[:, 2] * num_levels))

    raw_map = np.zeros(map.shape[:-1] + (2,))
    raw_map[threats[:, 0].astype(int), threats[:, 1].astype(int)] = np.stack((threat_levels, threats[:, 3]), axis=-1)
    raw_map[:, :, 0] -= (map.take(0, axis=-1) == -1)

    typed_threat_levels = np.ceil(np.expand_dims(threats[:, 2], axis=-1) * threat_matrix[threats[:, 3].astype(int)] * num_levels)
    typed_threat_matrix = np.zeros(map.shape)
    typed_threat_matrix[threats[:, 0].astype(int), threats[:, 1].astype(int)] = typed_threat_levels

    area_list = []

    # Build a list of areas
    # for t in range(num_levels+1):
    for i in range(map.shape[0]):
        for j in range(map.shape[1]):
            if (raw_map[i, j, 0] != -1) and (np.mod(raw_map[i, j, 0], 1) == 0):
                area_list.append(Area(raw_map, typed_threat_matrix[i, j], i, j))
    
    # Mark on a map which cell belongs to which area
    areas_map = np.empty_like(map.take(0, axis=-1), dtype=np.object0)
    for i in range(map.shape[0]):
        for j in range(map.shape[1]):
            for k in area_list:
                if (i, j) in k.cells:
                    areas_map[i, j] = k
                    break

    return area_list, areas_map

def assign_robot_area(area: Area, robot: Robot):
    area.assigned.append(robot)
    robot.area = area

def assign_areas_to_robots(areas: list[Area], areas_map: np.ndarray, robots: list[Robot], max_density: int):
    # Assign areas to robots
    for robot in robots:
        costs = robot.find_best_path_to_areas(areas) ####! HERE I STOPPED TO EXTEND TO HETEROGENEOUS !####
        for i in np.argsort(costs):
            area = areas[i]
            if len(area.assigned) < max_density * len(area.cells):
                assign_robot_area(area=area, robot=robot)
                break

    # Split areas with more than one robot assigned
    for area in areas.copy():
        if len(area.assigned) > 1:
            subareas = area.split_area_between_robots(map, areas_map)
            costs = np.array([robot.find_best_path_to_areas(subareas) for robot in area.assigned])

            for r, k in zip(range(len(area.assigned)), linear_sum_assignment(costs)[1]):
                assign_robot_area(subareas[k], area.assigned[r])
            areas.remove(area)
            areas = subareas + areas

    # Initialize robots' paths to the assigned areas
    #* This is a PATCH, because when area is splited the coverage status isn't preserved. It is fixable but I don't want to, for now
    #* KNOWN BUG: area of size 2 with 2 robots, after sharing paths are empty
    for robot in robots:
        if sum(robot.area.cells.values()) == len(robot.area.cells):
            areas.remove(areas_map[tuple(robot.location)])
            assign_next_area(areas, areas_map, robot)
        else:
            plan_robot_path(robot)
    
    return areas

def assign_next_area(areas: list[Area], areas_map: np.ndarray, robot: Robot):
    avail_areas = list(filter(lambda area: not area.assigned, areas))
    
    # There is an unassigned area, assign to the robot
    if avail_areas:
        if robot.graph_function == "OPTIMIZED":
            candidate_areas = avail_areas
        else:
            lowest_threat = min([area.threat_level[robot.type] for area in avail_areas])
            candidate_areas = list(filter(lambda area: area.threat_level[robot.type] == lowest_threat, avail_areas))
        
        costs = robot.find_best_path_to_areas(candidate_areas)
        area = candidate_areas[costs.argmin()]
        assign_robot_area(area, robot)
        plan_robot_path(robot)

    # No unassigned areas - find area to share
    else:
        avail_areas = areas.copy()
        for area in areas:
            robot.calc_path_to_area(area=area, areas_map=areas_map)
            if len(robot.path) > len(area.cells) - area.covered_cells: #* In the original algorithm it's ">" rather than ">="
                avail_areas.remove(area)
        
        for area in avail_areas.copy():
            robot.calc_path_to_area(area=area, areas_map=areas_map)
            print(f"{area}\tPath: {len(robot.path)}\tCells: {len(area.cells) - area.covered_cells}")


        if not avail_areas:
            robot.area = None
            return False

        costs = robot.find_best_path_to_areas(avail_areas)
        area = avail_areas[costs.argmin()]
        assign_robot_area(area, robot)

        # Split the area between the robots
        subareas = area.reallocate_area(areas_map)
        if len(subareas) == 1:
            subareas[0].assigned = area.assigned.copy()
            subareas = subareas[0].split_area_between_robots(map, areas_map)
        areas += subareas.copy()
        areas.remove(area)
        
        # Area cannot be splited (generally because only one cell remained)
        if len(subareas) == 1:
            robot.area = None
            assign_robot_area(subareas[0] ,area.assigned[0])
            return False

        for robot in area.assigned:
            costs = robot.find_best_path_to_areas(subareas)
            new_area = subareas[costs.argmin()]
            assign_robot_area(new_area, robot)
            subareas.remove(new_area)


        # Resets robots' paths to the assigned areas
        for robot in area.assigned:
            plan_robot_path(robot)

    return True

def plan_robot_path(robot: Robot):
    robot.calc_path_to_area(areas_map)
    robot.status = robot.STATUS["TRAVELING"] if len(robot.path) else robot.STATUS["COVERING"]
    if robot.status == robot.STATUS["COVERING"]:
        robot.calc_path_in_area(areas_map)


def init_env(areas: list[Area], areas_map: np.ndarray, robots: list[Robot], cover_map: np.ndarray):
    for robot in robots:
        area_status = areas_map[tuple(robot.location)].mark_as_covered(tuple(robot.location))
        if area_status:
            areas.remove(areas_map[tuple(robot.location)])
        cover_map[tuple(robot.location)] = 1

def make_parser():
    parser = ArgumentParser(description="Arguments for graph creation")
    parser.add_argument('map', type=str, nargs='?', default=None)
    parser.add_argument('graph_function', type=str, nargs='?', default=None)
    parser.add_argument('optim_alpha', type=float, nargs='?', default=None)
    return parser

if __name__ == '__main__':
    # Reading configuration file
    args = yaml.safe_load(open(os.path.join(os.getcwd(), "src", "classic", "config", "classic_default.yaml"), 'r'))
    parsed_args = make_parser().parse_args()

    for arg in args:
        if (arg in parsed_args) and (parsed_args.__getattribute__(arg)):
            args[arg] = parsed_args.__getattribute__(arg)
    config = Namespace(**args)
    
    results = [] # Store results of runs
    for _ in range(config.test_nepisode):
        print(f"### Initializing run {_}/{config.test_nepisode} ###")
        try:
            # Initialization
            map, robots, threats, threat_matrix = config2map(config)
            cover_map = np.zeros(map.shape[:-1], dtype=np.int16) + (map.take(0, axis=-1) == -1).astype(np.int16)

            # Pre-processing of the map
            areas_list, areas_map = areas_creation(map, threats, threat_matrix, num_levels=config.threat_levels)                # Split env into areas
            init_env(areas_list, areas_map, robots, cover_map)                                          # Mark robots locations as covered
            areas_list = assign_areas_to_robots(areas_list, areas_map, robots, config.max_density)      # Assign areas to robots
            
            # Running loop
            time = 0

            while True: # update
                #$ DEBUG
                print(f'Time: {time}\t{[[r.id for r in a.assigned] for a in areas_list]}')
                print(f'Time: {time}\t{[r.status for r in robots]}')
                
                # Perform action for every ACTIVE robot
                for robot in robots:
                    assert all([(areas_map[e] is not None) for e in robot.path]), "Invalid path"
                    if robot.status != robot.STATUS["DISABLED"]:
                        n_robots = sum([robot.status != robot.STATUS["DISABLED"] for robot in robots])
                        robot.create_induced_grid(map.take(robot.type, axis=-1), n_robots, cover_map)
                        robot_status, area_status = robot.step(map, areas_map)

                        if area_status: # Area is completely covered - removed from areas list
                            if areas_map[tuple(robot.location)] in areas_list:
                                print(f'Area is completed\tRobot:{robot.id}\t\tAssigned: ' + \
                                    f'{areas_map[tuple(robot.location)].assigned[0].id if areas_map[tuple(robot.location)].assigned else []}')
                                areas_list.remove(areas_map[tuple(robot.location)])

                        if robot_status == robot.STATUS["IDLE"]: # Robot finished to cover an area
                            flag = assign_next_area(areas_list, areas_map, robot) # Assign new area
                            # if robot.area:
                            #     print(f"AN AREA ASSIGNED WITH THREAT {map[list(robot.area.cells.keys())[0]][robot.type]} OUT OF {[map[list(area.cells.keys())[0]][robot.type] for area in areas_list if not area.assigned]}")

                        elif robot_status == robot.STATUS["DISABLED"] and (not area_status):
                            areas_list += robot.area.reallocate_area(areas_map)
                            areas_list.remove(robot.area)
                    
                    # Update measurements
                    cover_map[tuple(robot.location)] += 1 #$ DEBUG TO COUNT VISITS
                    assert all([len(area.assigned) < 2 for area in areas_list])
                
                # Update measurements
                time += 1

                if ((cover_map > 0).sum() == cover_map.size) or (time == config.episode_limit) or all([r.status == r.STATUS["DISABLED"] for r in robots]):
                    break # End condition

            optim_value = ((cover_map > 0) * (map.take(0, axis=-1) != -1)).sum() - config.optim_alpha * time # higher is better
            results.append({
                "coverage": ((cover_map > 0) * (map.take(0, axis=-1) != -1)).copy(),
                "time": time,
                "optim_value": optim_value
            })
        except:
            pass

    print_results(os.path.join(os.getcwd(), "src", "classic", "results"), config, results)
import numpy as np
from mrac_utils import *

# Aid constants
LEVELS = 4      # No. tiers of threats
DENSITY = 4     # Maximal ratio between robots and area cells


#! DOESN'T FIT TO HETEROGENOUS CASE!@!
#!      The area creator keep only one threat level

"""
data structures:
* the grid, described as 2D array, -1.0 = obstacle, 0.0 <= value <= 1.0 the risk in the cell
* a list of all connected areas in the grid, associated with their risk level
* management of all the robots (class?)
* 

$ General notes about the algorithm:
1. Non-optimality of the first assignment because it follows an arbitrary order of the robots
! 2. Consider to adopt the algorithm with the new (better) criteria by changing the heuristic of the induced graph
! 3. Handle case of area with size 1 and robot starts in this area
"""


def areas_creation(map, num_levels):
    raw_map = np.ceil((map * num_levels))
    area_list = []

    for t in range(num_levels):
        for i in range(map.shape[0]):
            for j in range(map.shape[1]):
                if raw_map[i, j] == t:
                    area_list.append(Area(raw_map, i, j, t))
    return area_list

def assign_robot_area(area: Area, robot: Robot):
    area.assigned.append(robot)
    robot.area = area

def assign_areas_to_robots(areas: list[Area], robots: list[Robot]):
    # Assign areas to robots
    for robot in robots:
        costs = robot.find_best_path_to_areas(areas)
        for i in np.argsort(costs):
            area = areas[i]
            if len(area.assigned) < DENSITY * len(area.cells):
                assign_robot_area(area=area, robot=robot)
                break

    # Split areas with more than one robot assigned
    for area in areas:
        area.split_area_between_robots()


if __name__ == '__main__':
    # Initialization
    map, robots = config2map("Random_Map3")

    # Pre-processing of the map
    areas_list = areas_creation(map, num_levels=LEVELS)
    assign_areas_to_robots(areas_list, robots)

    # Running loop
    time = 0
    while True: # update
        break # update
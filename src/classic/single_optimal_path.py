import numpy as np
import os, sys, yaml
sys.path.append('/home/ylab/experiments/MAPS/src')
from utils.dict2namedtuple import convert

MAP_PATH = os.path.join(os.getcwd(), 'maps', 'coverage_maps')
ACTIONS  = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])

def hash_state(agent_loc, map):
    return hash(agent_loc.__str__() + map.__str__())

def path_finder(map, alpha):
    # Get input
    path = os.path.join(MAP_PATH, map + '.yaml')
    with open(path, "r") as stream:
        config = yaml.safe_load(stream)
        config = convert(config)
    
    # Check validity
    assert config.n_agents == 1, "Invalid number of agents, this function works only for single agent"
    assert ~config.torodial, "This function is not designed to handle torodial environments"

    # Build the map with padding (does not change the optimal solution)
    grid = np.zeros(config.world_shape)
    
    obs = np.array(config.obstacles_location) # place obstacles
    grid[obs[:, 0], obs[:, 1]] = -1
    free_squares = grid.size - obs.shape[0] # Number of free squares for evaluating criteria

    threats = np.array(config.threat_location) # place threats
    grid[threats[:, 0].astype(int), threats[:, 1].astype(int)] = threats[:, 2]

    grid = np.pad(grid, 1, constant_values=-1) # the perimeter

    agent_init = np.array(config.agents_placement[0])


    # Start the scanning, using DFS
    # State defintion: (agent location, grid *expected* coverage, expected live, time steps)
    i_state = {
        'loc': agent_init,
        'cover': np.zeros(config.world_shape),
        'expect': 1 - grid[agent_init[0] + 1, agent_init[1] + 1],
        'time': 0
    }
    i_state['cover'][agent_init[0], agent_init[1]] = 1 # (agent placement is marked as covered)

    # Set of observed states for avoiding repitions
    observed = set()
    observed.add(hash_state(i_state['loc'], i_state['cover']))

    # Best criteria
    best_criteria = -np.inf
    best_state = None

    s = [i_state, ] # DFS stack
    while s:
        state = s.pop(-1)
        
        # Find available actions in state
        agent_locs = state['loc'] + ACTIONS
        avail_actions = grid[agent_locs[:, 0] + 1, agent_locs[:, 1] + 1] != -1

        for i in range(len(ACTIONS)):
            if avail_actions[i]:
                new_state = state.copy()

                agent_loc = agent_locs[i]
                if new_state['cover'][agent_loc[0], agent_loc[1]] < new_state['expect']: # expected coverage
                    new_state['cover'][agent_loc[0], agent_loc[1]] = new_state['expect'] 
                
                # if finished (all covered or the agent was disabled), save the value of criteria
                if ((new_state['cover'] > 0).sum() == free_squares) or (new_state['expect'] < 1e-3):
                    criteria = np.sum(new_state['cover']) - alpha * new_state['time']
                    if criteria > best_criteria:
                        best_criteria, best_state = criteria, state

                hash_value = hash_state(agent_loc, new_state['cover'])
                if hash_value in observed:
                    continue
                observed.add(hash_value)

                new_state['expect'] *= (1 - grid[agent_loc[0] + 1, agent_loc[1] + 1])
                new_state['time'] += 1

                s.append(new_state)

    return best_criteria, best_state


if __name__ == '__main__':
    # map = input("Enter map name: ") #Random_Single_Map1
    path_finder("Random_Single_Map1", alpha=0.5) #! SET ALPHA!
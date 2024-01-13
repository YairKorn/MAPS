import os
import numpy as np
np.set_printoptions(precision=2)
from collections import deque as queue

class map_generator():
    action_effect = np.asarray([[0, 1], [1, 0], [0, -1], [-1, 0], [0, 0]], dtype=np.int16)

    def __init__(
            self, path, height: int, width: int, n_agents: int,
            obstacle_rate=0.2, threats_rate=0.2, risk_avg=0.2, risk_std=0.2
    ):
        # Self properties for compatability
        self.height = height
        self.width = width
        self.n_agents = n_agents
        self.n_cells = height * width
        self.path = path

        # Generate map
        obs_map = self._place_obstacles(obstacle_rate)
        tht_map = self._place_threats(obs_map, threats_rate, risk_avg, risk_std)
        agents  = self._place_agents(obs_map)
        self._generate_file(obs_map, tht_map, agents)

    def _place_agents(self, obs_map):
        # Random placement - select random free cells for initiating the agents
        avail_cell = np.asarray(obs_map != -1, dtype=np.int16) # free cells
        l_cells = np.random.choice(self.n_cells, self.n_agents, replace=False, p=avail_cell.reshape(-1)/np.sum(avail_cell))
        return np.stack((l_cells/self.width, l_cells%self.width)).transpose().astype(np.int16)


    def _place_obstacles(self, obstacle_rate, num_trials=25):
        # Try to locate obstacles in the grid such that it remain reachable
        while True:
            for _ in range(num_trials):
                # Place obstacles randomly in the grid
                map_grid = -1 * (np.random.rand(self.height, self.width) < obstacle_rate)
                if -1 * np.sum(map_grid) + self.n_agents >= map_grid.size: # not enough place for the agents 
                    continue

                # Check that the obstacles distribution is valid
                test_grid = np.pad(map_grid, 1, constant_values=-1)
                root = np.stack(np.where(test_grid != -1)).transpose()[0, :]
                test_grid[root[0], root[1]] = 1 # mark root as visited

                q = queue() # queue for BFS (find if the grid is reachable)
                q.append(root)
                while q:
                    cell = q.popleft()
                    neighbors = cell + self.action_effect[:4]
                    neighbors = neighbors[test_grid[neighbors[:, 0], neighbors[:, 1]] == 0] # neighbors that are free (no obstacle) and wasn't visited
                    
                    test_grid[neighbors[:, 0], neighbors[:, 1]] = 1 # mark as visited
                    list(map(q.append, neighbors)) # add free-cell neighbors into queue

                if np.sum(np.absolute(test_grid[1:-1, 1:-1])) == self.n_cells: # obstacles (-1) + reachable cells (1) == grid size, i.e., the grid is reachable
                    return map_grid

            obstacle_rate *= 0.9 # reduce the obstacle rate because the current rate does not allow to create reachable environment
            print(f"INFO: Cannot create reachable environemnt, reduce obstacle rate to {obstacle_rate}")

    def _place_threats(self, obstacles, threats_rate, risk_avg, risk_std):
        avail_cell = np.asarray(obstacles != -1, dtype=np.int16) # free cells
        
        # normalize the threat_rate s.t. rate is relative to free cells and not the whole grid
        map_grid = avail_cell * (np.random.rand(self.height, self.width) < threats_rate * (avail_cell.size/np.sum(avail_cell)))
        map_grid = np.maximum(np.minimum(map_grid * np.random.normal(loc=risk_avg, scale=risk_std, size=map_grid.shape), 1), 0) # place threats using truncated normal distribution
        return map_grid


    def _generate_file(self, obstacles, threats, agents):
        file_str = "# A map for advarserial coverage environment\n\n"
        
        file_str += "agents_placement: "
        file_str += agents.tolist().__str__()
        
        file_str += "\nn_agents: "
        file_str += str(agents.shape[0])

        file_str += "\nobstacles_location: "
        obstacles_location = np.vstack(np.where(obstacles == -1)).T
        # file_str += obstacles_location.__str__().replace(' ', ', ').replace('\n', '')
        file_str += obstacles_location.tolist().__str__()

        file_str += "\nthreat_location: "
        threats_location = np.vstack(np.where(threats > 0))
        threats_location = np.vstack((threats_location, threats[threats_location[0, :], threats_location[1, :]])).T
        file_str += threats_location.tolist().__str__().replace("], ", "],\n")

        file_str += "\ntorodial: False"
        file_str += "\nworld_shape: "
        file_str += [self.height, self.width].__str__()

        file_str += "\nrandom_placement: False"
        with open(self.path, 'w') as f:
            f.write(file_str)
        

if __name__ == '__main__':
    for i in [20]:
        wsize = np.random.randint(20, 21)
        n_agents = 15 #int(wsize / 3) + np.random.randint(1, 4)

        path = f'/home/ylab/experiments/MAPS/maps/coverage_maps/Random_Single_Map{i+1}.yaml'
        if os.path.exists(path):
            ans = input("File already exists, do you want to override? Y/n\n")
            if ans != 'Y':
                print(f"Skip file: {path}")
                continue

        map_generator(path=path,
            height=wsize, width=wsize, n_agents=n_agents,
            obstacle_rate=0.25 * np.random.rand() + 0.15, threats_rate=0.3 * np.random.rand(),
             risk_avg=0.3 * np.random.rand() + 0.10, risk_std=0.2 * np.random.rand() + 0.2)
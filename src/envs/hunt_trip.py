"""
ENV DESCRIPTION: 
This environment simulates the MRAC (multi-robot adversarial coverage), i.e. coverage with threatened areas.
The environment includes n robots located in grid of MxN cells, and each cell should be visited by a robot at least once.

CONFIGURATION:
The simulator can create an environment based on some key parameters, or use pre-defined configuration.
In addition, the simulator can create new environment every episode for "meta-learning" (enabling "shuffle_config").
In that case, "watch_surface" flag must be enabled, otherwise the environment violates the Markov principle.


NOTES:
* The environment does not include the extension for heterogeneous robots
* Environment is not compatable for batch mode for now (lean code without extra features)
? enable random placement of agents only once?
"""
import os, yaml, datetime
from envs.multiagentenv import MultiAgentEnv
from utils.dict2namedtuple import convert
from collections import deque as queue
import numpy as np
import matplotlib.pyplot as plt
MAP_PATH = os.path.join(os.getcwd(), 'maps', 'hunt_trip_maps')

class HuntingTrip(MultiAgentEnv):

    action_labels = {'right': 0, 'down': 1, 'left': 2, 'up': 3, 'stay': 4, 'catch': 5}
    action_effect = np.asarray([[0, 1], [1, 0], [0, -1], [-1, 0], [0, 0], [0, 0]], dtype=np.int16)

    def __init__(self, **kwargs):
        # Unpack arguments from sacred
        args = kwargs
        
        # Unpack environment configuration from map file
        env_map = os.path.join(MAP_PATH, (args['map'] if args['map'] is not None else 'default') + '.yaml')
        with open(env_map, "r") as stream:
            map_config = yaml.safe_load(stream)
            print(map_config)
            for k, v in map_config.items():
                args[k] = v
        
        # convert to GenericDictionary
        if isinstance(args, dict):
            args = convert(args)
        self.args = args

        # Set up the environment
        world_shape = np.asarray(args.world_shape, dtype=np.int16) 
        self.height, self.width = world_shape
        self.toroidal = getattr(args, "toroidal", False)
        self.allow_collisions = getattr(args, "allow_collisions", False)
        self.catch_validity = getattr(args, "catch_validity", False)
        self.n_actors = np.asarray([args.n_agents, args.n_preys], dtype=np.int16)
        
        # Initialization
        np.random.seed = getattr(args, "random_seed", None) # set seed for numpy
        self.grid = np.zeros((self.height, self.width, 4), dtype=np.float16)
        self.n_cells = self.grid[:, :, 0].size
        # grid structure: agents locations | preys locations | surface
        
        self.shuffle_config = getattr(args, "shuffle_config", False)
        self.obstacle_rate  = getattr(args, "obstacle_rate", 0.2)

        # place obstacles in the area
        if not self.shuffle_config:
            if getattr(args, "random_config", False): # random configuration - only once
                self.grid[:, :, 2] = self._place_obstacles(self.obstacle_rate)
                self.obstacles = np.stack(np.where(self.grid[:, :, 2] == -1)).transpose()
            else: # place ocnfiguration from configuration file
                self.obstacles = np.asarray(getattr(args, "obstacles_location", []))
                if self.obstacles.size > 0:
                    self.grid[self.obstacles[:, 0], self.obstacles[:, 1], 2] = -1

        self.episode_limit = args.episode_limit
        
        # Observation properties
        self.observe_state = getattr(args, "observe_state", False)
        self.observe_ids = getattr(args, "observe_ids", False)
        self.watch_carried = getattr(args, "watch_carried", True)
        self.watch_surface = getattr(args, "watch_surface", True)
        self.observation_range = getattr(args, "observation_range", -1) # -1 = full observability
        self.n_features = 3 + self.watch_surface + self.watch_carried # changable features, does not includes the location
        
        # For partial-observable environment, observation is r in each direction, resulting a (2r+1) square around the agent
        self.state_size = self.grid.size
        self.obs_size = self.n_features * (self.n_cells if self.observation_range < 0 else (2 * self.observation_range + 1) ** 2)
        
        #! Should I give an option for exact loaction?

        # Agents' action space
        self.n_actions = 6 # 5 move directions + 1 catch action
        self.failure_prob = getattr(args, "failure_prob", 0.0)
        self.avail_actions = np.pad(self.grid[:, :, 2], 1, constant_values=(self.toroidal - 1))        
        
        self.actor_placement  = np.zeros((2, self.n_actors.max(), 2), dtype=np.int16)
        self.random_placement = np.zeros(2, dtype=np.int16)

        # Set initial locations of actors
        try:
            for i, arg in zip(range(2), ["agents_placement", "preys_placement"]):
                arr = np.asarray(getattr(args, arg, []))
                if arr.size:        
                    self.actor_placement[i, :self.n_actors[i]] = arr[:self.n_actors[i]]
                else:
                    self.random_placement[i] = 1
        except:
            raise "Failed to locate actors in environment, check configuration file"

        #! Sanity check for actors location
        # if (self.agents_placement.shape[0] and self.agents_placement.shape[0] < self.n_agents) or \
        #     (self.preys_placement.shape[0] and self.preys_placement.shape[0] < self.n_agents):
        #     raise ValueError("Failed to locate all actors in the grid")
        # if any(self.grid[self.agents_placement[:, 0], self.agents_placement[:, 1], 1] == -1) or \
        #     any(self.grid[self.preys_placement[:, 0], self.preys_placement[:, 1], 1] == -1):
        #     raise ValueError("Actors cannot be located on obstacles, check out the configuration file")

        # Reward function
        self.reward_hunt      = getattr(args, "reward_hunt", 6.0)
        self.reward_catch     = getattr(args, "reward_catch", -1.0)
        self.reward_carry     = getattr(args, "reward_carry", -0.2)
        self.reward_move      = getattr(args, "reward_move", -0.4)
        self.reward_stay      = getattr(args, "reward_stay", -0.1)
        self.reward_collision = getattr(args, "reward_collision", 0.0)

        # Internal variables
        self.actors = np.zeros_like(self.actor_placement, dtype=np.int16)
        self.prey_for_agent = np.zeros(self.n_actors[0], dtype=np.int16)
        self.prey_available = np.ones(self.n_actors[1], dtype=np.int16)

        # Array for calculating reward for moves
        self.action_reward  = np.asarray([self.reward_move] * 4 + [self.reward_stay] + [self.reward_catch], dtype=np.float16)
        
        self.steps = 0
        self.reset()

    ################################ Env Functions ################################
    def reset(self, **kwargs):
        # Reset old episodes - preys, grid & statistisc
        self.prey_available.fill(1)
        self.prey_for_agent.fill(0)
        self.grid[:, :, :2].fill(0.0)
        self.grid[:, :, 3].fill(0.0)
        self.steps = 0

        # If "shuffle_config" mode in on, the area is changed every episode (obstacles)
        if self.shuffle_config:
            self.grid[:, :, 2] = self._place_obstacles(self.obstacle_rate)
            self.obstacles = np.stack(np.where(self.grid[:, :, 2] == -1)).transpose()

        # Place agents & preys
        for i in range(2):
            self._place_actors(actor=i)

    # "invalid_agents", "collision" allow decomposition of the reward per agent -
    # wasn't implemented for compatability reasons
    def step(self, actions):
        actions = np.asarray(actions.cpu(), dtype=np.int16)
        if actions.size != self.n_actors[0]:
            raise ValueError("Wrong number of actions")
        
        # Randomly (w.p. failure_prob) an action fails
        actions[np.where((np.random.random(self.n_actors[0]) < self.failure_prob) * (actions < 5))] = 4 #! CHECK

        # Basic reward for actions: basic reward for actions + negative reward for carring preys
        reward = self.action_reward[actions].sum() + self.reward_carry * (self.prey_for_agent * (actions != self.action_labels["stay"])).sum()
        
        # Move the agents (avoid collision if allow_collision is False)
        new_locations = self._enforce_validity(0, self.action_effect[actions] + self.actors[0, :self.n_actors[0]])
        for agent in np.random.permutation(self.n_actors[0]):
            if not np.array_equal(new_locations[agent], self.actors[0, agent]):
                self.grid[self.actors[0, agent, 0], self.actors[0, agent, 1], 3] = 0
                collision = self._move_actor(0, agent, new_locations[agent])
                self.grid[self.actors[0, agent, 0], self.actors[0, agent, 1], 3] = self.prey_for_agent[agent]

                reward += collision * self.reward_collision

            if actions[agent] == self.action_labels["catch"]:
                adjacent_preys = (np.absolute(self.actors[1] - new_locations[agent]).sum(axis=1) <= 1) * (self.prey_available)
                
                if adjacent_preys.sum():
                    hunted_prey = np.random.choice(self.n_actors[1], p=adjacent_preys/adjacent_preys.sum()) # randomly select a prey to catch
                    self.grid[self.actors[1, hunted_prey, 0], self.actors[1, hunted_prey, 1], 1] = 0
                    self.grid[self.actors[0, agent, 0], self.actors[0, agent, 1], 3] += 1
                    self.prey_for_agent[agent] += 1
                    self.prey_available[hunted_prey] = 0
                    
                    reward += self.reward_hunt
        
        # Move the preys (random in the grid)
        preys_actions = self.action_effect[np.random.choice(5, size=self.n_actors[1])] * self.prey_available.reshape(-1, 1) # 5 actions - 4 moves + stay, with uniform distribution
        new_locations = self._enforce_validity(1, preys_actions + self.actors[1, :self.n_actors[1]])
        for prey in np.random.permutation(self.n_actors[1]):
            if not np.array_equal(new_locations[prey], self.actors[1, prey]):
                self._move_actor(1, prey, new_locations[prey])

        self.steps += 1

        # if (all preys were caught) or (episode limit was reached), end the episode
        terminated = (not np.sum(self.grid[:, :, 1])) or self.steps >= self.episode_limit
        info = {"episode_limit": self.steps >= self.episode_limit}

        return reward, terminated, info

    # Claculate avaiable actions for all the agents
    def get_avail_actions(self):
        return [self.get_avail_agent_actions(agent) for agent in range(self.n_actors[0])]

    # Calculate the available actions for a specific agent
    def get_avail_agent_actions(self, agent):
        next_location = self.actors[0, agent] + self.action_effect + 1 # avail_actions is padded
        avail_actions = self.avail_actions[next_location[:, 0], next_location[:, 1]] != -1

        if not self.allow_collisions:
            next_location = self.actors[0, agent] + avail_actions.reshape(-1, 1) * self.action_effect
            avail_actions[:4] = self.grid[next_location[:4, 0], next_location[:4, 1], 0] == 0
        
        if self.catch_validity:
            avail_actions[self.action_labels["catch"]] = (((self.actors[1] - self.actors[0, agent]).abs().sum(axis=1) <= 1) * (self.prey_available)).any()
        
        return avail_actions

    def close(self):
        print("Closing Hunting Trip Environment")

    ################################ Obs Functions ################################
    def get_obs(self):
        agents_obs = [self.get_obs_agent(i) for i in range(self.n_actors[0])]
        return agents_obs

    # Return the full state (privileged knowledge)
    def get_state(self):
        return (self.observe_state * self.grid.copy()).reshape(self.state_size)

    # There are 2 observation modes:
    #   1. Fully-observable (the whole area is observed) - observation_range = -1
    #   2. Partial-observability with absolute location (location of agent is marked on the map)
    #   3. Partial-observability without absolute location
    def get_obs_agent(self, agent_id):
        # Filter the grid layers that available to the agent
        watch = np.unique([0, 1, 2 * self.watch_surface, 3 * self.watch_carried])

        # Observation-mode 1 - return the whole grid
        if self.observation_range < 0:
            observation = self.grid[:, :, watch].copy()
            agent_location = (self.actors[0, agent_id, 0], self.actors[0, agent_id, 1])

        # Observation-mode 2&3 - return a (2*range + 1)x(2*range + 1) slice from the grid
        else:
            d = 2*self.observation_range + 1
            observation = np.dstack((np.zeros((d, d)), np.ones((d, d)), -1*np.ones((d, d))))
            pass #! complete to partial-observability with absolute location -> mask the invisible parts of the state

        # Remove agents' id from observation if defined
        if not self.observe_ids: 
            observation[:, :, 0] = (observation[:, :, 0] != 0)

        # Add agent's location (one-hot)
        one_hot = np.expand_dims(np.zeros_like(observation[:, :, 0]), axis=2)
        one_hot[agent_location[0], agent_location[1], 0] = 1.0

        # First layer is one-hot of agent's location, the other are the state data, reshaped to 1D vector
        return np.dstack((one_hot, observation)).reshape(-1)

    def get_state_size(self):
        return self.state_size

    def get_total_actions(self):
        return self.n_actions

    def get_obs_size(self):
        return self.obs_size

    def get_env_info(self):
        info = MultiAgentEnv.get_env_info(self)
        return info

    def get_stats(self):
        pass

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_actors[0],
                    "episode_limit": self.episode_limit}
        return env_info


    ################################ Internal Functions ################################
    def _enforce_validity(self, a_type, new_locations):
        # Enforce border validity
        if self.toroidal:
            new_locations %= self.grid.shape[:2]
        else:
            new_locations = np.stack([np.maximum(np.minimum(new_locations[:, i], self.grid.shape[i] - 1), 0) for i in [0, 1]], axis=1)

        # Enforce obstacle validity
        into_obstacle = np.where(self.grid[new_locations[:, 0], new_locations[:, 1], 2] == -1)[0]
        new_locations[into_obstacle] = self.actors[a_type, into_obstacle]

        return new_locations


    def _move_actor(self, a_type, a_id, new_location):
        # Check for collisions
        if (not self.allow_collisions) and (self.grid[new_location[0], new_location[1], :(a_type+1)] != 0).all():
            return True
        
        # Move the actors
        self.grid[self.actors[a_type, a_id, 0], self.actors[a_type, a_id, 1], a_type] = 0.0
        self.actors[a_type, a_id] = new_location
        self.grid[self.actors[a_type, a_id, 0], self.actors[a_type, a_id, 1], a_type] = a_id * (1 - a_type) + 1.0
        return False

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

    def _place_actors(self, actor):
        # Calculate placement
        if self.random_placement[actor]:
            avail_cell = np.asarray((self.grid != 0).sum(axis=2) == 0, dtype=np.int16)
            l_cells = np.random.choice(self.n_cells, self.n_actors[actor], replace=False, p=avail_cell.reshape(-1)/np.sum(avail_cell))
            placement = np.stack((l_cells/self.width, l_cells%self.width)).transpose().astype(np.int16)
        else:
            placement = self.actor_placement[actor]

        # Locate actors in the grid and store their location
        self.grid[placement[:self.n_actors[actor], 0], placement[:self.n_actors[actor], 1], actor] = \
            (1 if actor else (np.arange(self.n_actors[actor]) + 1))
        self.actors[actor, :self.n_actors[actor]] = placement

    def _calc_placement(self):
        # Calculate the first n_agent empty cells in the grid
        return np.stack(np.where(self.grid[:, :, 2] != -1)).transpose()[:self.n_agents]

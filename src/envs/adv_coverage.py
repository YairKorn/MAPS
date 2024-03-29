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
import json
import os, yaml, datetime, imageio
from envs.multiagentenv import MultiAgentEnv
from utils.dict2namedtuple import convert
from collections import deque as queue
import numpy as np
np.set_printoptions(precision=2)

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", module="matplotlib\..*" )

MAP_PATH = os.path.join(os.getcwd(), 'maps', 'coverage_maps')


class AdversarialCoverage(MultiAgentEnv):

    action_labels = {'right': 0, 'down': 1, 'left': 2, 'up': 3, 'stay': 4}
    action_effect = np.asarray([[0, 1], [1, 0], [0, -1], [-1, 0], [0, 0]], dtype=np.int16)

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
        self.n_agents = args.n_agents

        # Initialization
        np.random.seed = getattr(args, "random_seed", None) # set seed for numpy
        self.grid = np.zeros((self.height, self.width, 4), dtype=np.float16)
        self.n_cells = self.grid[:, :, 0].size
        # grid structure: agent locations | coverage status | surface
        
        self.shuffle_config = getattr(args, "shuffle_config", False)
        self.obstacle_rate  = getattr(args, "obstacle_rate", 0.2)
        self.threats_rate   = getattr(args, "threats_rate", 0.2)
        self.risk_avg       = getattr(args, "risk_avg", 0.2)
        self.risk_std       = getattr(args, "risk_std", 0.2)
        self.fix_seed       = getattr(args, "fix_seed", False)
        self.reps_per_seed  = getattr(args, "reps_per_seed", 1)
        self.tests_till_now = 0

        # place obstacles and threats in the area
        if not self.shuffle_config:
            if getattr(args, "random_config", False): # random configuration - only once
                self.grid[:, :, 2] = -1 * self._place_obstacles(self.obstacle_rate)
                self.grid[:, :, 3] += self._place_threats(self.threats_rate, self.risk_avg, self.risk_std)
                self.obstacles = np.stack(np.where(self.grid[:, :, 2] == 1)).transpose()
            else: # place ocnfiguration from configuration file
                self.obstacles = np.asarray(getattr(args, "obstacles_location", []))
                if self.obstacles.size > 0:
                    self.grid[self.obstacles[:, 0], self.obstacles[:, 1], 2] = 1

                threats = np.asarray(getattr(args, "threat_location", []))
                if threats.size > 0:
                    threats_location = np.asarray(threats[:, :2], dtype=np.int16)
                    self.grid[threats_location[:, 0], threats_location[:, 1], 3] = threats[:, 2]

        self.episode_limit = args.episode_limit
        self.reduced_punish = getattr(args, "reduced_punish", 0.0) # never disable a robot but give a negative reward for entering a threat
        self.reduced_decay = getattr(args, "reduced_decay", False)
        self.reduced_shape = getattr(args, 'reduced_shape', 'decay')
        self.simulated_mode = getattr(args, 'simulated_mode', False)

        # Observation properties
        self.observe_state = getattr(args, "observe_state", False)
        self.observe_ids = getattr(args, "observe_ids", False)
        self.watch_covered = getattr(args, "watch_covered", True)
        self.watch_surface = getattr(args, "watch_surface", True)
        self.observation_range = getattr(args, "observation_range", -1) # -1 = full observability
        self.n_features = 1 + self.watch_covered + 2 * self.watch_surface # changable features, does not includes the location
        
        self.state_size = self.grid.size
        self.obs_size = self.n_cells + self.n_features * (self.n_cells if self.observation_range < 0 else (2 * self.observation_range + 1) ** 2)
        
        # Agents' action space
        self.allow_stay = getattr(args, "allow_stay", True)
        self.n_actions = 5 #4 + self.allow_stay
        self.avail_actions = np.pad(self.grid[:, :, 2], 1, constant_values=(self.toroidal - 1))
        self.random_placement = getattr(args, "random_placement", True) # random placements of the robots every episode, override "agent_placement"
        
        placements = np.asarray(getattr(args, "agents_placement", []))   # otherwise, can select a specific setup
        self.agents_placement = placements[:self.n_agents] if placements.size > 0 else self._calc_placement()
        
        # Sanity-checking for agents location
        if not self.random_placement:
            if self.agents_placement.shape[0] < self.n_agents:
                raise ValueError("Failed to locate all agents in the grid")
            if any(self.grid[self.agents_placement[:, 0], self.agents_placement[:, 1], 2] == -1):
                raise ValueError("Agents cannot be located on obstacles, check out the configuration file")

        # Reward function
        self.time_reward      = getattr(args, "reward_time", -0.1)
        self.collision_reward = getattr(args, "reward_collision", 0.0)
        self.new_cell_reward  = getattr(args, "reward_cell", 1.0)
        self.succes_reward    = getattr(args, "reward_succes", 0.0)
        self.invalid_reward   = getattr(args, "reward_invalid", 0.0)
        self.threat_reward    = getattr(args, "reward_threat", 1.0) # this constant is multiplied by the designed reward function

        # Logging
        self.log_env       = getattr(args, "log_env", False)
        self.log_stat      = {
             "covered": np.zeros(world_shape),
             "episode": 0
        }
        self.log_collector = []
        self.test_mode     = False
        self.nepisode      = 0
        self.evaluate = False
        self.log_id = getattr(args, "id", '0')

        # Internal variables
        self.agents = np.zeros(shape=(self.n_agents, 2), dtype=np.int16)
        self.agents_enabled = np.ones(self.n_agents, dtype=np.int16)
        self.steps = 0
        self.sum_rewards = 0
        self.reset()

    ################################ Env Functions ################################
    def reset(self, **kwargs):
        # Reset old episodes - enable agents, clear grid, reset statistisc
        self.agents_enabled.fill(1)
        self.grid[:, :, 0:2].fill(0.0)
        self.steps = 0
        self.sum_rewards = 0

        # If "shuffle_config" mode in on, the area is changed every episode (obstacles and threats)
        if self.shuffle_config:
            self.grid[:, :, 2] = -1 * self._place_obstacles(self.obstacle_rate)
            self.grid[:, :, 3] += self._place_threats(self.threats_rate, self.risk_avg, self.risk_std)
            self.obstacles = np.stack(np.where(self.grid[:, :, 2] == 1)).transpose()

        # self.threat_factor = 1.0
        if kwargs:
            self.test_mode = kwargs['test_mode'] or kwargs['evaluate']
            self.nepisode  = kwargs['test_nepisode']
            self.evaluate = kwargs['evaluate']

            # PUNISH FACTOR gradually increases the threats in the area
            if not self.test_mode:
                if self.reduced_shape == 'decay':
                    self.threat_factor = min(1 - (1.0 - self.reduced_punish) * (1 - kwargs['t_env'] / (kwargs['t_max'] * self.args.reduced_decay)), 1.0) \
                        if self.args.reduced_decay > 0 else self.reduced_punish
                elif self.reduced_shape == 'cutoff':
                    self.threat_factor = self.reduced_punish if (kwargs['t_env'] / kwargs['t_max']) < self.args.reduced_decay else 1.0
                else:
                    raise "Invalid reduced reward function"
            else:
                self.threat_factor = 1.0
            self.simulated_mode = getattr(self.args, 'simulated_mode', False) * (not self.test_mode)
            self.succes_reward = getattr(self.args, 'reward_succes', False) * (not self.test_mode)
            # print(f"Threat factor: {self.threat_factor}")

        # Place agents & set obstacles to marked as "covered"
        self.agents = self._place_agents()
        self.grid[self.agents[:, 0], self.agents[:, 1], 0] = (np.arange(self.n_agents) + 1)
        self.grid[self.agents[:, 0], self.agents[:, 1], 1] = 1
        if self.obstacles.size > 0:
            self.grid[self.obstacles[:, 0], self.obstacles[:, 1], 1] = 1


    # "invalid_agents", "collision" allow decomposition of the reward per agent -
    # wasn't implemented for compatability reasons
    def step(self, actions):
        actions = np.asarray(actions.cpu(), dtype=np.int16)
        if actions.size != self.n_agents:
            raise ValueError("Wrong number of actions")
        actions[np.where(self.agents_enabled == 0)] = self.action_labels["stay"]  # mask actions of disabled agents (disabled -> stay)
        
        # rewards for agents
        reward = self.time_reward

        # Calculate theoretical new locations
        new_locations, invalid_agents = self._enforce_validity(self.action_effect[actions] + self.agents)
        reward += invalid_agents.size * self.invalid_reward

        # Move the agents (avoid collision if allow_collision is False)
        for agent in np.random.permutation(self.n_agents):
            if not np.array_equal(new_locations[agent], self.agents[agent]): # the agent should move
                new_cell, collision = self._move_agent(agent, new_locations[agent])
                
                # calculate reward for the step - includes whether a new cell covered & if collision occurred
                reward += new_cell * self.new_cell_reward + (collision is not None) * self.collision_reward

        # Threats reward - indicate for the agents that they are in danger
        e = np.where(self.agents_enabled == 1)[0] # enabled agents
        total_threats = np.sum(self.grid[self.agents[e, 0], self.agents[e, 1], 3])
        covered = np.sum(self.grid[:, :, 1])
        alive_agents = e.size

        reward += (1 + self.time_reward) * (total_threats * (self.n_cells - covered) * (self.time_reward/(alive_agents) if alive_agents > 1 else -1))

        # Apply risks in area on the agents (disable robots w.p. associated to the cell)
        # threat_effect = np.random.random(self.n_agents) > self.grid[self.agents[:, 0], self.agents[:, 1], 3] * (1 - self.simulated_mode)
        threat_effect = np.random.random(self.n_agents) > self.grid[self.agents[:, 0], self.agents[:, 1], 3] * self.threat_factor

        temp_agent_enabled = self.agents_enabled.copy()
        self.agents_enabled *= threat_effect

        d = temp_agent_enabled != self.agents_enabled # agents disabled right now are removed from the grid
        self.grid[self.agents[d, 0], self.agents[d, 1], 0] = 0

        mission_succes = np.sum(self.grid[:, :, 1]) == self.n_cells
        reward += mission_succes * self.succes_reward
        if mission_succes:
            print(f"Mode: {'Test    ' if self.test_mode else 'Training'} | Coverage was completed! | Reward={reward}")
        
        self.steps += 1
        self.sum_rewards += reward

        # if the whole area was covered, or all agents are disabled, or episode limit was reached, end the episode
        terminated = mission_succes or not np.sum(self.agents_enabled) or self.steps >= self.episode_limit
        info = {"episode_limit": self.steps >= self.episode_limit}

        # Logging (only for test episodes, when logger is on)
        if self.log_env and self.test_mode and terminated:
            self._logger()
        
        if self.fix_seed and self.test_mode and terminated:
            self.tests_till_now += 1

        return reward, terminated, info

    # Calculate available actions for all the agents
    def get_avail_actions(self):
        return [self.get_avail_agent_actions(agent) for agent in range(self.n_agents)]

    # Calculate the available actions for a specific agent
    def get_avail_agent_actions(self, agent):
        if not self.agents_enabled[agent]:
            avail_actions = np.array([False] * 4 + [True])
        else:
            next_location = self.agents[agent] + self.action_effect + 1 # avail_actions is padded
            avail_actions = self.avail_actions[next_location[:, 0], next_location[:, 1]] != -1

            next_location = self.agents[agent] + avail_actions.reshape(-1, 1) * self.action_effect
            avail_actions = self.grid[next_location[:, 0], next_location[:, 1], 0] == 0
            avail_actions[-1] = self.allow_stay or (not self.agents_enabled[agent])
        
        return avail_actions

    def close(self):
        if self.args.visualize:
            self._visualize_learning(frame_rate=self.args.frame_rate)
        print("Closing MRAC Environment")

    ################################ Obs Functions ################################
    def get_obs(self):
        agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        return agents_obs

    # Return the full state (privileged knowledge)
    def get_state(self):
        observation = self.grid.copy()
        # if not self.observe_ids: # remove agents' id from observation
        #     observation[:, :, 0] = (observation[:, :, 0] != 0)
        
        return observation.reshape(self.state_size)

    # There are 2 observation modes:
    #   1. Fully-observable (the whole area is observed) - observation_range = -1
    #   2. Partial-observability with absolute location (location of agent is marked on the map)
    def get_obs_agent(self, agent_id):
        # Filter the grid layers that available to the agent
        watch = np.unique([0, self.watch_covered, 2 * self.watch_surface, 3 * self.watch_surface])

        # Observation-mode 1 - return the whole grid
        if self.observation_range < 0:
            observation = self.grid[:, :, watch].copy()
            agent_location = (self.agents[agent_id, 0], self.agents[agent_id, 1])

        # Observation-mode 2 - return a (2*range + 1)x(2*range + 1) slice from the grid
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

    ################################ Internal Functions ################################
    def _enforce_validity(self, new_locations):
        agents_invalid = np.empty(0) # list of agents that tries to perform an invalid action
        
        # Enforce border validity
        if self.toroidal:
            new_locations %= self.grid.shape[:2]
        else:
            temp_loc = np.stack([np.maximum(np.minimum(new_locations[:, i], self.grid.shape[i] - 1), 0) for i in [0, 1]], axis=1)
            agents_invalid = np.where(temp_loc != new_locations)[0]
            new_locations = temp_loc

        # Enforce obstacle validity
        into_obstacle = np.where(self.grid[new_locations[:, 0], new_locations[:, 1], 2] == -1)[0]
        new_locations[into_obstacle] = self.agents[into_obstacle]

        # return new locations & list of agents tried to preform invalid actions
        return new_locations, np.concatenate((agents_invalid, into_obstacle))


    def _move_agent(self, agent, new_location):
        # Check for collisions
        if (not self.allow_collisions) and (self.grid[new_location[0], new_location[1], 0] != 0):
            return False, np.asarray([agent, self.grid[new_location[0], new_location[1], 0] - 1])
        
        # Move the agent
        self.grid[self.agents[agent, 0], self.agents[agent, 1], 0] = 0.0
        self.agents[agent] = new_location
        self.grid[self.agents[agent, 0], self.agents[agent, 1], 0] = agent + 1.0
        
        # Mark the new cell as covered
        new_cell = self.grid[self.agents[agent, 0], self.agents[agent, 1], 1] == 0.0
        self.grid[self.agents[agent, 0], self.agents[agent, 1], 1] = 1.0
        return new_cell, None

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

    def _place_threats(self, threats_rate, risk_avg, risk_std):
        avail_cell = np.asarray(self.grid[:, :, 2] != 1, dtype=np.int16) # free cells
        
        # normalize the threat_rate s.t. rate is relative to free cells and not the whole grid
        map_grid = avail_cell * (np.random.rand(self.height, self.width) < threats_rate * (avail_cell.size/np.sum(avail_cell)))
        map_grid = np.maximum(np.minimum(map_grid * np.random.normal(loc=risk_avg, scale=risk_std, size=map_grid.shape), 1), 0) # place threats using truncated normal distribution
        return map_grid

    def _place_agents(self):
        # Random placement - select random free cells for initiating the agents
        if self.random_placement:
            avail_cell = np.asarray(self.grid[:, :, 2] != 1, dtype=np.int16)  # free cells

            # seed = int(self.tests_till_now / self.reps_per_seed) if (self.test_mode and self.fix_seed) else None
            seed = np.random.randint(0, 5)
            random_generator = np.random.RandomState(seed)
            l_cells = random_generator.choice(self.n_cells, self.n_agents, replace=False, p=avail_cell.reshape(-1)/np.sum(avail_cell))
            return np.stack((l_cells/self.width, l_cells%self.width)).transpose().astype(np.int16)

        # Else, set the agents in the set placement (random placement override setted placements)
        return self.agents_placement.copy()

    def _calc_placement(self):
        # Calculate the first n_agent empty cells in the grid
        return np.stack(np.where(self.grid[:, :, 2] != -1)).transpose()[:self.n_agents]

    # Simple logger to track agents' performances
    def _logger(self):
        self.log_stat["episode"] += 1
        self.log_stat["covered"] += self.grid[:, :, 1]

        if self.log_stat["episode"] == self.nepisode:
            self.log_collector.append(self.log_stat["covered"].copy() / self.log_stat["episode"])
            avg_cells = (np.sum(self.log_stat["covered"])) / self.log_stat["episode"] - len(self.obstacles)
            print('ENV LOG | Average cells covered: {:.2f} out of {} free cells'.format(avg_cells, self.n_cells-len(self.obstacles)))
            print(str(self.log_stat["covered"] * (self.grid[:, :, 2] != -1) / self.log_stat["episode"] - (self.grid[:, :, 2] == -1)).replace('-1. ', '  XX'))
            
            self.log_stat["covered"] *= 0
            self.log_stat["episode"] = 0

    # Visualize dymanic of learning over time
    def _visualize_learning(self, frame_rate=0.2):
        FIG_SIZE = 6

        # Set up directory for saving results
        result_path = os.path.join(os.getcwd(), "results", "sacred", self.log_id, "env_log")
        os.makedirs(result_path)

        # Print threats on the map
        textmap = [[str(self.grid[i, j, 3]) if self.grid[i, j, 3] > 0 else "" for j in range(self.width)] for i in range(self.height)]
        steps_pad = int(np.log10(len(self.log_collector))) + 1

        # Per frame, restore the relevant stat and draw it using colormap
        for s in range(len(self.log_collector)):
            # Calculate data for table
            f = self.log_collector[s]
            colormap = [[self.__colormap((i, j), f[i, j]) for j in range(self.width)] for i in range(self.height)]
            
            # Create and configure plot
            fig, ax = plt.subplots()
            ax.set_title(f'MRAC With Map={self.args.map}, K={self.n_agents}; Time={str(s).zfill(steps_pad)}', \
                fontsize='x-large', fontweight='bold', fontname='Ubuntu', pad=15)
            fig.set_figheight(FIG_SIZE*0.95)
            fig.set_figwidth(FIG_SIZE)

            the_table = ax.table(cellText=textmap, cellColours=colormap, cellLoc='center', loc='center')
            cell_height = 1 / self.height
            for _, cell in the_table.get_celld().items():
                cell.set_height(cell_height)
            ax.axis("off")
            
            plt.savefig(os.path.join(result_path, 'fig' + str(s).zfill(steps_pad) + '.png'))
            plt.close()

        images = [imageio.imread(os.path.join(result_path, image)) for image in sorted(os.listdir(result_path))]
        imageio.mimsave(os.path.join(result_path, 'visualization.gif'), images, duration=frame_rate)

    # An aid-function for the visualizer
    def __colormap(self, p, v):
        x, y = p
        return (0.0, 0.0, 0.0) if (self.grid[x, y, 2] == -1) else (1.0-0.411*v, 0.588+0.411*v, 0.588)
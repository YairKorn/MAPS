"""
ENV DESCRIPTION: 
This environment simulates the MRAC (multi-robot adversarial coverage), i.e. coverage with threatened areas.
The environment includes n robots located in grid of MxN cells, and each cell should be visited by a robot at least once.



NOTES:
* The environment does not include the extension for heterogeneous robots
* Environment is not compatable for batch mode for now (lean code without extra features)

! what about disabling STAY action?
"""

from numpy.core.fromnumeric import shape
from torch._C import int16
from envs.multiagentenv import MultiAgentEnv
from utils.dict2namedtuple import convert
import numpy as np


class StagHunt(MultiAgentEnv):

    action_labels = {'stay': 0, 'right': 1, 'down': 2, 'left': 3, 'up': 4}

    def __init__(self, **kwargs):
        # Unpack arguments from sacred
        args = kwargs
        if isinstance(args, dict):
            args = convert(args)
        self.args = args

        # Initialization
        np.random.seed(getattr(args, "random_seed", None)) # set seed for numpy
        self.grid = np.zeros((self.height, self.width, 3), dtype=np.float16)
        # grid structure: agent locations | coverage status | surface
        
        self.obstacles = getattr(args, "obstacles_location", []) # locating obstacles on the map
        self.grid[self.obstacles[:, 0], self.obstacles[:, 1], 2] = -1
        
        threats   = getattr(args, "threat_location",    []) # locating threats on the map
        self.grid[threats[:, 0], threats[:, 1]] = threats[:, 2]

        self.episode_limit = args.episode_limit
        self.simulated_mode   = getattr(args, "simulated_mode", False) # never disable a robot but give a negative reward for entering a threat
        
        # Observation properties
        self.observe_ids = getattr(args, "observe_ids", False)
        self.self_location = getattr(args, "self_location", True)
        self.watch_covered = getattr(args, "watch_covered", False)
        self.watch_surface = getattr(args, "watch_surface", False)
        self.observation_range = getattr(args, "observation_range", -1) # -1 = full observability
        self.n_feats = 1 + self.self_location + self.watch_covered + self.watch_surface # features per cell
        
        self.state_size = self.grid.size
        #! state representation - agent-centric or general???

        # Build the environment
        world_shape = np.asarray(args.world_shape, dtype=np.int16) 
        self.height, self.width = world_shape
        self.toroidal = getattr(args, "toroidal", False)
        self.allow_collisions = getattr(args, "allow_collisions", False)

        # Agents' action space
        self.n_agents = args.n_agents
        self.n_actions = 5
        self.action_effect = np.asarray([[0, 0], [0, 1], [1, 0], [0, -1], [-1, 0]], dtype=np.int16)
        self.random_placement = getattr(args, "random_placement", True) # random placements of the robots every episode, override "agent_placement"
        
        placements = getattr(args, "agents_placement", np.empty())   # otherwise, can select a specific setup
        self.agents_placement = placements[:self.n_agents] if placements.size > 0 else self._calc_placement()
        if self.agents_placement.shape[0] < self.n_agents:
            raise ValueError("Failed to locate all agents in the grid")

        # Reward function
        self.time_reward      = getattr(args, "reward_time", -0.1)
        self.collision_reward = getattr(args, "reward_collision", 0.0)
        self.new_cell_reward  = getattr(args, "reward_cell", 1.0)
        self.succes_reward    = getattr(args, "succes_reward", 0.0)
        self.invalid_reward   = getattr(args, "invalid_reward", 0.0)
        self.threat_reward    = getattr(args, "threat_reward", 1.0) # this constant is multiplied by the designed reward function

        # Internal variables
        self.agents = np.zeros(shape=(self.n_agents, 2), dtype=np.int16)
        self.agents_enabled = np.ones(self.n_agents, dtype=np.int16)
        self.steps = 0
        self.sum_rewards = 0
        self.reset()

    ################################ Env Functions ################################
    def reset(self):
        # Reset old episodes - enable agents, clear grid, reset statistisc
        self.agents_enabled.fill(1)
        self.grid[:, :, 0:2].fill(0.0)
        self.steps = 0
        self.sum_rewards = 0

        # Place agents & set obstacles to marked as "covered"
        self.agents = self._place_agents()
        self.grid[self.obstacles[:, 0], self.obstacles[:, 1], 1] = 1
        #! return self.get_obs(), self.get_state() #* check this out!

    # "invalid_agents", "collision" allow decomposition of the reward per agent -
    # wasn't implemented for compatability reasons
    def step(self, actions):
        if actions.size != self.n_agents:
            raise ValueError("Wrong number of actions")
        actions *= self.agents_enabled # mask actions of disabled agents (disabled -> stay)
        
        # rewards for agents
        reward = self.time_reward

        # Calculate theoretical new locations
        new_locations, invalid_agents = self._enforce_validity(self.action_effect[actions] + self.agents)
        reward += invalid_agents.size * self.invalid_reward

        # Move the agents (avoid collision if allow_collision is False)
        for agent in np.random.permutation(self.n_agents):
            if new_locations[agent] != self.agents[agent]: # the agent should move
                new_cell, collision = self._move_agent(agent, new_locations[agent])
                
                # calculate reward for the step - includes whether or not a new cell covered & if collision occured
                reward += new_cell * self.new_cell_reward + collision * self.collision_reward

        # Threats reward - indicate for the agents that they are in danger
        total_threats = np.sum(self.grid[self.agents[0, self.alive_agents], self.agents[1, self.alive_agents], 2])
        remaining_cells = self.grid[:, :, 1].size - np.sum(self.grid[:, :, 1])
        alive_agents = np.sum(self.agents_enabled)

        reward += (total_threats * remaining_cells * (self.time_reward if alive_agents > 1 else 1)) / alive_agents

        # Apply risks in area on the agents (disable robots w.p. associated to the cell)
        threat_effect = np.random.random(self.n_agents) > self.grid[self.agents[0, :], self.agents[1, :], 2]
        self.agents_enabled *= threat_effect

        mission_succes = np.sum(self.grid[:, :, 1] == self.grid[:, :, 1].size)
        reward += mission_succes * self.succes_reward

        self.steps += 1
        self.sum_rewards += reward

        # if the whole area was covered, or all agents are disabled, or episode limit was reached, end the episode
        terminated = mission_succes or not np.sum(self.agents_enabled) or self.steps >= self.episode_limit
        info = {"episode_limit": terminated}
        return reward, terminated, info

    def get_env_info(self):
        info = MultiAgentEnv.get_env_info(self)
        return info

    ################################ Obs Functions ################################
    def get_obs(self):
        agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        return agents_obs

    def get_state(self):
        return self.grid.copy().reshape(self.state_size)

    ################################ Internal Functions ################################
    def _enforce_validity(self, new_locations):
        agents_invalid = np.empty(0) # list of agents that tries to perform an invalid action
        
        # Enforce border validity
        if self.toroidal:
            new_locations %= self.grid.shape[:2]
        else:
            temp_loc = np.maximum(np.minimum(new_locations, self.env_max - 1), 0)
            agents_invalid = np.where(temp_loc != new_locations)[0]
            new_locations = temp_loc

        # Enforce obstacle validity
        into_obstacle = np.where(self.grid[new_locations[:, 0], new_locations[:, 1]] == -1)
        new_locations[into_obstacle] = self.agents[into_obstacle]

        # return new locations & list of agents tried to preform invalid actions
        return new_locations, np.concatenate(agents_invalid, into_obstacle)


    def _move_agent(self, agent, new_location):
        # Check for collisions
        if self.allow_collisions or self.grid[new_location[0], new_location[1], 0] != 0:
            return False, np.asarray([agent, self.grid[new_location[0], new_location[1], 0] - 1])
        
        # Move the agent
        self.grid[self.agents[agent[0]], self.agents[agent[1]], 0] = 0.0
        self.agents[agent] = new_location
        self.grid[self.agents[agent[0]], self.agents[agent[1]], 0] = agent + 1.0
        
        # Mark the new cell as covered
        new_cell = self.gragents_enabledid[self.agents[agent[0]], self.agents[agent[1]], 1] == 0.0
        self.grid[self.agents[agent[0]], self.agents[agent[1]], 1] = 1.0
        return new_cell, None

    def _place_agents(self):
        # Random placement - select random free cells for initiating the agents
        if self.random_placement:
            avail_cell = np.asarray(self.grid[:, :, 2] != -1, dtype=np.int16) # free cells
            l_cells = np.random.choice(self.grid[:, :, 2].size, self.n_agents, replace=False, p=avail_cell.reshape(-1)/sum(avail_cell))
            return np.stack((l_cells/self.width, l_cells%self.width)).astype(np.int16)

        # Else, set the agents in the set placement (random placement override setted placements)
        return self.agents_placement.copy()

    def _calc_placement(self):
        # Calculate the first n_agent empty cells in the grid
        return np.stack(np.where(self.grid[:, :, 2] != -1)).transpose()[:self.n_agents]
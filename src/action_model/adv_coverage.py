import os, yaml
import numpy as np
import torch as th
from .model_v1 import ActionModel
from collections import defaultdict
MAP_PATH = os.path.join(os.getcwd(), 'maps', 'coverage_maps')

class AdvCoverage(ActionModel):
    def __init__(self, scheme, args) -> None:
        super().__init__(scheme, args)

        # Basics of action model
        self.action_effect = th.tensor([[0, 1], [1, 0], [0, -1], [-1, 0], [0, 0]], dtype=th.int)
        self.state_repr = defaultdict(lambda: ' ')
        for k ,v in [(9.0, 'X'), (1.0, '*'), (-1.0, '$'), (2.0, '#')]:
            self.state_repr[k] = v
        # self.state_repr = {0.0: ' ', 9.0: 'X', 1.0: '*', -1.0: '$', 2.0: '#'}

        #! PyMARL doesn't support user-designed maps so it's a little bit artificial here
        env_map = os.path.join(MAP_PATH, (self.args.map if self.args.map is not None else 'default') + '.yaml')
        with open(env_map, "r") as stream:
            self.height, self.width = yaml.safe_load(stream)['world_shape']
            self.n_cells = self.height * self.width

        # Observation properties
        self.watch_covered = getattr(self.args, "watch_covered", True)
        self.watch_surface = getattr(self.args, "watch_surface", True)
        self.observation_range = getattr(self.args, "observation_range", -1) # -1 = full observability

        # Reward function
        self.time_reward      = getattr(self.args, "reward_time", -0.1)
        self.collision_reward = getattr(self.args, "reward_collision", 0.0)
        self.new_cell_reward  = getattr(self.args, "reward_cell", 1.0)
        self.succes_reward    = getattr(self.args, "reward_succes", 0.0)
        self.invalid_reward   = getattr(self.args, "reward_invalid", 0.0)
        self.threat_reward    = getattr(self.args, "reward_threat", 1.0) # this constant is multiplied by the designed reward function

        self.agents = th.zeros((self.n_agents, 2), dtype=th.long) # location vector
        self.enable = self.n_agents * [True]

    """ When new perception is percepted, update the real state """
    def _update_env_state(self, state):
        self.state = state.reshape(self.height, self.width, -1)

        # extract agents' locations from the state
        temp_agents = th.stack(th.where(self.state[:, :, 0] > 0)).transpose(0, 1).cpu()
        identities = self.state[temp_agents[:, 0], temp_agents[:, 1], 0].long() - 1
        self.agents[identities] = temp_agents

        # extract agents' status from state
        self.enable = [agent_id in identities for agent_id in range(self.n_agents)]
    
    """ Use the general state to create an observation for the agents """
    def get_obs_agent(self, agent_id):
        # Filter the grid layers that available to the agent
        watch = np.unique([0, self.watch_covered, 2 * self.watch_surface])

        # Observation-mode 1 - return the whole grid
        observation = self.state[:, :, watch].clone()
        agent_location = (self.agents[agent_id, 0], self.agents[agent_id, 1])

        # Remove agents' id from observation - #* A robot in threatened cell is (1-p) alive
        observation[:, :, 0] = (observation[:, :, 0] != 0)
        observation[self.agents[:, 0], self.agents[:, 1], 0] *= 1 - self.state[self.agents[:, 0], self.agents[:, 1], 2]
        self.alive = (observation[:, :, 0] > 0).sum()

        # Add agent's location (one-hot)
        one_hot = th.unsqueeze(th.zeros_like(observation[:, :, 0]), axis=2)
        one_hot[agent_location[0], agent_location[1]] = 1.0

        # First layer is one-hot of agent's location, the other are the state data, reshaped to 1D vector
        return th.dstack((one_hot, observation)).reshape(-1)

    # Use avail_actions to detect obstacles; this function avoid collisions (case that another agent has moved to adjacent cell)
    def get_avail_actions(self, agent_id, avail_actions):
        new_loc = self.agents[agent_id] + self.action_effect * avail_actions.transpose(0, 1).cpu()
        no_collision = (self.state[new_loc[:, 0], new_loc[:, 1], 0] != 0) * \
            (self.state[new_loc[:, 0], new_loc[:, 1], 0] != (agent_id + 1))
        return avail_actions * (~ no_collision)


    """ Simulate the result of action in the environment """
    def _apply_action_on_state(self, agent_id, action, avail_actions, result=0):
        if not self.enable[agent_id]: # if agents is disabled, do nothing
            terminated = (self.t == self.episode_limit - 1) or (not sum(self.enable))
            return self.time_reward / self.n_agents, terminated
        
        agent_location = self.agents[agent_id, :]
        if not avail_actions.flatten()[action]:
            action = 4 # Do nothing (does not support invalid/collision reward)

        # apply agent movement
        new_location = agent_location + self.action_effect[action]
        new_location %= th.tensor(self.state.shape[:2])

        if self.state[new_location[0], new_location[1], 0] == 0 and self.state[new_location[0], new_location[1], 2] >= 0:
            self.state[agent_location[0], agent_location[1], 0] = 0.0
            self.state[new_location[0], new_location[1], 0] = agent_id + 1.0
            self.agents[agent_id, :] = new_location

        # mark new cell as covered
        new_cell = self.state[new_location[0], new_location[1], 1] == 0.0
        self.state[new_location[0], new_location[1], 1] = 1.0

        # check termination
        covered = th.sum(self.state[:, :, 1])
        terminated = (covered == self.height * self.width) or (self.t == self.episode_limit - 1) or (not sum(self.enable))

        # calculate reward
        reward = self.time_reward / self.n_agents + self.new_cell_reward * new_cell                 # Time & Cover
        reward += self.state[new_location[0], new_location[1], 2] * (self.n_cells - covered) * \
            (self.time_reward if self.alive > 1 else -1) / self.alive                               # Threats

        return reward, terminated
    
    def _back_update(self, obs, state, ind):
        obs = obs.reshape(self.height, self.width, -1)
        for agent_id in self.action_order[:ind]:
            obs[self.agents[agent_id, 0], self.agents[agent_id, 1], 1] = self.enable[agent_id]

        return obs.reshape(-1)

    #$ DEBUG: plot a transition specified by time, including state, action, reward and new state
    def plot_transition(self, t, bs=0):
        # print selected action
        action = self.buffer["actions"][bs, t, 0, 0]
        avail_actions = th.where(self.buffer["avail_actions"][bs, t, 0, :])[0].cpu().tolist()
        reward = self.buffer["reward"][bs, t, 0].item()
        print(f'Action: {action}\t Available Actions: {avail_actions}\t Reward: {reward}')
        
        state = self.buffer["obs"][bs, t, 0, :].reshape(self.height, self.width, -1)
        state = (state[:, :, 1] * (1 - 3*state[:, :, 0]) + state[:, :, 2] - 8*state[:, :, 3]).cpu().tolist()
        state = [[self.state_repr[e] for e in row] for row in state]

        new_state = self.buffer["obs"][bs, t+1, 0, :].reshape(self.height, self.width, -1)
        new_state = (new_state[:, :, 1] * (1 - 3*new_state[:, :, 0]) + new_state[:, :, 2] - 8*new_state[:, :, 3]).cpu().tolist()
        new_state = [[self.state_repr[e] for e in row] for row in new_state]

        for i in range(len(state)):
            print(str(state[i]).replace("'","") + '\t' + str(new_state[i]).replace("'",""))
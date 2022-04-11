import os, yaml
import numpy as np
import torch as th
from .basic_model import BasicAM
from collections import defaultdict
MAP_PATH = os.path.join(os.getcwd(), 'maps', 'hunt_trip_maps')

class HuntingTrip(BasicAM):
    def __init__(self, scheme, args) -> None:
        #! PyMARL doesn't support user-designed maps so it's a little bit artificial here
        env_map = os.path.join(MAP_PATH, (args.env_args["map"] if args.env_args["map"] is not None else 'default') + '.yaml')
        with open(env_map, "r") as stream:
            self.height, self.width = yaml.safe_load(stream)['world_shape']
            self.n_cells = self.height * self.width
        
        # Ideally, __init__ should've start here
        super().__init__(scheme, args, stochastic=True)

        # Basics of action model
        self.directed_catch = getattr(self.args, "directed_catch", True)
        self.catch_order   = th.tensor([[0, 0], [0, 1], [1, 0], [0, -1], [-1, 0]], dtype=th.int16)
        self.action_effect = th.tensor([[0, 1], [1, 0], [0, -1], [-1, 0], [0, 0]], dtype=th.int16)
        self.action_effect = th.cat((self.action_effect, th.zeros((1 + 4 * self.directed_catch, 2), dtype=th.int16)), axis=0)
        
        self.state_repr = defaultdict(lambda: ' ')
        for k ,v in [(-1.0, 'X'), (1.0, '+'), (3.0, '$'), (4.0, '$'), (2.0, '#')]:
            self.state_repr[k] = v

        # Observation properties
        self.watch_carried = getattr(self.args, "watch_carried", True)
        self.watch_surface = getattr(self.args, "watch_surface", True)
        self.observation_range = getattr(self.args, "observation_range", -1) # -1 = full observability

        # Reward function
        self.reward_hunt      = getattr(self.args, "reward_hunt", 6.0)
        self.reward_catch     = getattr(self.args, "reward_catch", -1.0)
        self.reward_carry     = getattr(self.args, "reward_carry", -0.2)
        self.reward_move      = getattr(self.args, "reward_move", -0.4)
        self.reward_stay      = getattr(self.args, "reward_stay", -0.1)
        self.reward_collision = getattr(self.args, "reward_collision", 0.0)
        self.action_reward  = th.tensor([self.reward_move] * 4 + [self.reward_stay] + (1 + 4 * self.directed_catch) * [self.reward_catch], dtype=th.float16)

        # Env-specific data structures
        self.failure_prob   = getattr(self.args, "failure_prob", 0.0)
        self.prey_for_agent = th.zeros(self.n_agents, dtype=th.int16) #! CONSIDER REMOVE?

        self.intended_locs = th.zeros((self.n_agents, 2), dtype=th.int16)

    """ When new perception is percepted, update the real state """
    def _update_env_state(self, state):
        state = state.reshape(self.height, self.width, -1)
        data = self.mcts_buffer.sample(take_one=True)
 
        # Update state information
        temp_agents = th.stack(th.where(state[:, :, 0] > 0)).transpose(0, 1).cpu()
        identities = state[temp_agents[:, 0], temp_agents[:, 1], 0].long() - 1

        data["state"][0] = state
        data["agents"][0][identities] = temp_agents
        data["carried"][0] = state[data["agents"][0, :, 0], data["agents"][0, :, 1], 3].reshape(-1, 1)

        return data
    
    """ Use the general state to create an observation for the agents """
    def get_obs_state(self, data, agent_id):
        # Filter the grid layers that available to the agent
        watch = np.unique([0, 1, 2 * self.watch_surface, 3 * self.watch_carried])
        state, agents = data["state"], data["agents"]

        # Observation mode 1 - return the whole grid
        observation = state[:, :, watch].clone()
        agent_location = (agents[agent_id, 0], agents[agent_id, 1])

        # Remove agents' id from observation
        observation[:, :, 0] = (observation[:, :, 0] != 0)

        # if not self.apply_MCTS:
        #     acted = th.tensor([agent_id in self.action_order for agent_id in range(self.n_agents)])
        #     observation[agents[:, 0], agents[:, 1], 0] *= 1 - state[agents[:, 0], agents[:, 1], 2] * acted

        # Add agent's location (one-hot)
        one_hot = th.unsqueeze(th.zeros_like(observation[:, :, 0]), axis=2)
        one_hot[agent_location[0], agent_location[1]] = 1.0
        # assert observation[:, :, 0].sum() == data["enable"].sum(), "Wrong update"

        # First layer is one-hot of agent's location, the other are the state data, reshaped to 1D vector
        return th.dstack((one_hot, observation)).reshape(-1)


    # Use avail_actions to detect obstacles; this function avoid collisions (case that another agent has moved to adjacent cell)
    def get_avail_actions(self, data, agent_id, avail_actions):
        # Sample the (arbitrary) first state option to calculate available actions
        new_loc = data["agents"][0, agent_id] + self.action_effect * avail_actions.transpose(0, 1).cpu()
        no_collision = (data["state"][0, new_loc[:, 0], new_loc[:, 1], 0] != 0) * \
            (data["state"][0, new_loc[:, 0], new_loc[:, 1], 0] != (agent_id + 1))
        return avail_actions * (~ no_collision)


    """ Simulate the result of action in the environment """
    def _apply_action_on_state(self, data, agent_id, action, avail_actions):
        state, agents, carried, result = data["state"], data["agents"], data["carried"], data["result"]
        agent_location = agents[agent_id, :].clone()

        # Calculate reward based on the action
        reward = self.action_reward[action] + (action != 4) * self.reward_carry * carried[agent_id]

        # No move is performed
        if (not avail_actions.view(-1)[action]):
            action = 4

        # Apply agent movement
        if action < 5:
            new_location = (agent_location + self.action_effect[action]) % th.tensor(state.shape[:2])
            self.intended_locs[agent_id] = new_location # keep the intended location of the agent

            if result == 0: # if move succeed (result == 0), update the state
                state[agent_location[0], agent_location[1], 0] = 0.0

                state[new_location[0], new_location[1], 0] = agent_id + 1.0
                agents[agent_id, :] = new_location

        # Apply "catch" action
        if (action > 4) and (result == 0):
            #! KNOWN PROBLEM: if two (or more) agents apply "catch" the action model doesn't know who will take the prey
            #!                for back-updating, one need to reorder the action selection
            #!                for now, ignoring it and try...
            prey_loc = self._select_prey(state, agent_location, action)
            if prey_loc is not None:
                state[prey_loc[0], prey_loc[1], 1] = 0
                self.prey_for_agent[agent_id] += 1
                reward = reward + self.reward_hunt
        
        return reward, th.sum(state[:, :, 1]) == 0
    
    """ back_update """
    def _back_update(self, batch, data, t, n_episodes):
        obs = batch["obs"][0, t, 0, :].view(self.height, self.width, -1)

        r = self.action_order[:n_episodes]
        for agent in r:
            if (data["agents"][0, agent] != self.intended_locs[agent]).all():
                obs[data["agents"][0, agent, 0], data["agents"][0, agent, 1], 1] = 1    # return the agent to its previous location
                obs[self.intended_locs[agent, 0], self.intended_locs[agent, 1], 1] = (data["agents"][0, r] == data["agents"][0, agent]).all(dim=1).any()

        # # Update the reward based on results
        # agent_id = self.action_order[n_episodes] # the agent we're updating
        # new_cell = 1 - obs[data["agents"][0][agent_id, 0], data["agents"][0][agent_id, 1], 2]
        # enable_agents = (data["enable"][0, self.action_order[:n_episodes]]).sum() + (self.prev_enable[self.action_order[n_episodes:]]).sum()

        # batch["reward"][0, t, 0] =  self.time_reward / self.n_agents + self.new_cell_reward * new_cell
        # batch["reward"][0, t, 0] += self.prev_enable[agent_id] * obs[data["agents"][0][agent_id, 0], data["agents"][0][agent_id, 1], 3] * \
        #     (self.n_cells - th.sum(obs[:, :, 2]) - new_cell) * (self.time_reward/(enable_agents) if enable_agents > 1 else -1)

        # # Update the termination status based on 
        # batch["terminated"][0, t, 0] = ((obs[:, :, 1].sum() == 0) or (obs[:, :, 2].sum() == self.n_cells) or (batch["terminated"][0, t, 0]))

        # assert obs[:, :, 1].sum() == data["enable"][0, self.action_order[:n_episodes]].sum() + self.prev_enable[self.action_order[n_episodes:]].sum(), "Wrong update"
        # if self.t - t == 1:
        #     self.prev_enable = data["enable"][0].clone()
        # return obs.reshape(-1)

    def _action_results(self, data, agent_id, action):
        if action < 4: # Move (but not "stay")
            return th.tensor([1.0-self.failure_prob, self.failure_prob])
        
        return th.tensor([1.0]) # "Stay" and "catch" always succeeds
        

    def detect_interaction(self, data):
        return np.arange(self.n_agents)


    def _get_mcts_scheme(self, scheme, args):
        return {
            "state": ((self.height, self.width, 4), th.float32),
            "hidden": ((args.rnn_hidden_dim, ), th.float32),
            "agents": ((args.n_agents, 2), th.long),
            "carried": ((args.n_agents, 1), th.long)
        }

    def _select_prey(self, state, cell, action):
        # Directed catch - check if there is a prey in the desired cell
        if self.directed_catch:
            test_loc = cell + self.action_effect[action - 5]
            if th.min(test_loc) < 0 or test_loc[0] >= state.shape[0] or test_loc[1] >= state.shape[1]:
                return None

            if state[test_loc[0], test_loc[1], 1]:
                return test_loc
        
        # Non-directed catch - select by catch_order
        else:
            for i in self.catch_order:
                test_loc = cell + i
                if not (th.min(test_loc) < 0 or test_loc[0] >= state.shape[0] or test_loc[1] >= state.shape[1]):
                    if state[test_loc[0], test_loc[1], 1]:            
                        return test_loc
        return None

    #$ DEBUG: plot a transition specified by time, including state, action, reward and new state
    def plot_transition(self, t, bs=0):
        # print selected action
        action = self.buffer["actions"][bs, t, 0, 0]
        avail_actions = th.where(self.buffer["avail_actions"][bs, t, 0, :])[0].cpu().tolist()
        reward = self.buffer["reward"][bs, t, 0].item()
        print(f'Action: {action}\t Available Actions: {avail_actions}\t Reward: {reward}')
        
        state = self.buffer["obs"][bs, t, 0, :].reshape(self.height, self.width, -1)
        state = (state[:, :, 0] + 2 * state[:, :, 1] + state[:, :, 2] + state[:, :, 3]).cpu().tolist()
        state = [[self.state_repr[e] for e in row] for row in state]

        new_state = self.buffer["obs"][bs, t+1, 0, :].reshape(self.height, self.width, -1)
        new_state = (new_state[:, :, 0] + 2 * new_state[:, :, 1] + new_state[:, :, 2] + new_state[:, :, 3]).cpu().tolist()
        new_state = [[self.state_repr[e] for e in row] for row in new_state]

        for i in range(len(state)):
            print(str(state[i]).replace("'","") + '\t' + str(new_state[i]).replace("'",""))
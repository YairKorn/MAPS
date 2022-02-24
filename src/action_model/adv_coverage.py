from cgitb import enable
import os, yaml
import numpy as np
import torch as th
from .basic_model import BasicAM
from collections import defaultdict
MAP_PATH = os.path.join(os.getcwd(), 'maps', 'coverage_maps')

class AdvCoverage(BasicAM):
    def __init__(self, scheme, args) -> None:
        #! PyMARL doesn't support user-designed maps so it's a little bit artificial here
        env_map = os.path.join(MAP_PATH, (args.env_args["map"] if args.env_args["map"] is not None else 'default') + '.yaml')
        with open(env_map, "r") as stream:
            self.height, self.width = yaml.safe_load(stream)['world_shape']
            self.n_cells = self.height * self.width
        
        # Ideally, __init__ should've start here
        super().__init__(scheme, args, stochastic=True)

        # Basics of action model
        self.action_effect = th.tensor([[0, 1], [1, 0], [0, -1], [-1, 0], [0, 0]], dtype=th.int)
        self.state_repr = defaultdict(lambda: ' ')
        for k ,v in [(9.0, 'X'), (1.0, '*'), (-1.0, '$'), (2.0, '#')]:
            self.state_repr[k] = v

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

        # Env-specific attributes
        self.prev_enable      = th.ones(self.n_agents)

    """ When new perception is percepted, update the real state """
    def _update_env_state(self, state):
        state = state.reshape(self.height, self.width, -1)
        data = self.mcts_buffer.sample(take_one=True)

        # Extract agents' locations from the state
        temp_agents = th.stack(th.where(state[:, :, 0] > 0)).transpose(0, 1).cpu()
        identities = state[temp_agents[:, 0], temp_agents[:, 1], 0].long() - 1
        assert (not self.t) or (data["agents"][0][identities] == temp_agents).all(), "Mismatch update!"
        
        # Update state information
        data["state"][0] = state
        data["agents"][0][identities] = temp_agents
        data["enable"][0] = th.tensor([agent_id in identities for agent_id in range(self.n_agents)])

        #$ TEST
        if not self.t:
            self.prev_enable = th.ones(self.n_agents, dtype=th.long)

        return data
    
    """ Use the general state to create an observation for the agents """
    def get_obs_state(self, data, agent_id):
        # Filter the grid layers that available to the agent
        watch = np.unique([0, self.watch_covered, 2 * self.watch_surface])
        state, agents = data["state"], data["agents"]

        # Observation-mode 1 - return the whole grid
        observation = state[:, :, watch].clone()
        agent_location = (agents[agent_id, 0], agents[agent_id, 1])

        # Remove agents' id from observation
        observation[:, :, 0] = (observation[:, :, 0] != 0)

        if not self.apply_MCTS:
            acted = th.tensor([agent_id in self.action_order for agent_id in range(self.n_agents)])
            observation[agents[:, 0], agents[:, 1], 0] *= 1 - state[agents[:, 0], agents[:, 1], 2] * acted

        # Add agent's location (one-hot)
        one_hot = th.unsqueeze(th.zeros_like(observation[:, :, 0]), axis=2)
        one_hot[agent_location[0], agent_location[1]] = 1.0
        assert observation[:, :, 0].sum() == data["enable"].sum(), "Wrong update"

        # First layer is one-hot of agent's location, the other are the state data, reshaped to 1D vector
        return th.dstack((one_hot, observation)).reshape(-1)


    # Use avail_actions to detect obstacles; this function avoid collisions (case that another agent has moved to adjacent cell)
    def get_avail_actions(self, data, agent_id, avail_actions):
        if not data["enable"][0, agent_id]:
            return avail_actions

        # Sample the (arbitrary) first state option to calculate available actions
        new_loc = data["agents"][0, agent_id] + self.action_effect * avail_actions.transpose(0, 1).cpu()
        no_collision = (data["state"][0, new_loc[:, 0], new_loc[:, 1], 0] != 0) * \
            (data["state"][0, new_loc[:, 0], new_loc[:, 1], 0] != (agent_id + 1))
        return avail_actions * (~ no_collision)


    """ Simulate the result of action in the environment """
    def _apply_action_on_state(self, data, agent_id, action, avail_actions):
        state, agents, result, enable = data["state"], data["agents"], data["result"], data["enable"]
        enable_agents = max(enable.sum(), 1) # number of enable agents, bound by 1 for prevent divergence
        agent_location = agents[agent_id, :].clone()

        # No move is performed
        if (not avail_actions.view(-1)[action]):
            action = 4

        if not enable[agent_id]:
            new_location = agent_location
        
        else:
            # Apply agent movement
            new_location = (agent_location + self.action_effect[action]) % th.tensor(state.shape[:2])
            state[agent_location[0], agent_location[1], 0] = 0.0

            enable[agent_id] *= (1 - result) # possible results: 0 = enabled, 1 = disabled
            state[new_location[0], new_location[1], 0] = (agent_id + 1.0) * enable[agent_id]
            agents[agent_id, :] = new_location

        # mark new cell as covered
        new_cell = state[new_location[0], new_location[1], 1] == 0.0
        state[new_location[0], new_location[1], 1] = 1.0

        assert (enable * (th.arange(self.n_agents) + 1)).sum() == state[:, :, 0].sum(), "Wrong update"

        # Termination status
        terminated = (th.sum(state[:, :, 1]) == self.n_cells) or (not sum(enable))

        return 0, terminated
    
    """ back_update """
    def _back_update(self, batch, data, t, n_episodes):
        obs = batch["obs"][0, t, 0, :].view(self.height, self.width, -1)

        r = self.action_order[:n_episodes]
        for agent in r:
            if data["enable"][0, agent] != self.prev_enable[agent]:
                cell_status = ((data["agents"][0, r] == data["agents"][0, agent]).all(dim=1) * data["enable"][0, r]).any()
                obs[data["agents"][0][agent, 0], data["agents"][0][agent, 1], 1] = cell_status

        # Update the reward based on results
        agent_id = self.action_order[n_episodes] # the agent we're updating
        new_cell = 1 - obs[data["agents"][0][agent_id, 0], data["agents"][0][agent_id, 1], 2]
        enable_agents = (data["enable"][0, self.action_order[:n_episodes]]).sum() + (self.prev_enable[self.action_order[n_episodes:]]).sum()

        batch["reward"][0, t, 0] =  self.time_reward / self.n_agents + self.new_cell_reward * new_cell
        batch["reward"][0, t, 0] += self.prev_enable[agent_id] * obs[data["agents"][0][agent_id, 0], data["agents"][0][agent_id, 1], 3] * \
            (self.n_cells - th.sum(obs[:, :, 2]) - new_cell) * (self.time_reward/(enable_agents) if enable_agents > 1 else -1)

        # Update the termination status based on 
        batch["terminated"][0, t, 0] = ((obs[:, :, 1].sum() == 0) or (obs[:, :, 2].sum() == self.n_cells) or (batch["terminated"][0, t, 0]))

        assert obs[:, :, 1].sum() == data["enable"][0, self.action_order[:n_episodes]].sum() + self.prev_enable[self.action_order[n_episodes:]].sum(), "Wrong update"
        if self.t - t == 1:
            self.prev_enable = data["enable"][0].clone()
        return obs.reshape(-1)

    def _action_results(self, data, agent_id, action):
        # Possible action results depends on the threat in the new location
        new_location = data["agents"][0, agent_id, :] + self.action_effect[action]
        p = data["state"][0, new_location[0], new_location[1], 2] * data["enable"][0, agent_id]
        return th.tensor([1.0-p, p] if p else [1.0])

    def detect_interaction(self, data):
        # Only enabled agents take an action; if all disabled, arbitrary select 0 to get the terminated state
        return np.arange(self.n_agents)

    def _get_mcts_scheme(self, scheme, args):
        return {
            "state": ((self.height, self.width, 3), th.float32),
            "hidden": ((args.rnn_hidden_dim, ), th.float32),
            "agents": ((args.n_agents, 2), th.long),
            "enable": ((self.n_agents, ), th.long)
        }

    #$ DEBUG: plot a transition specified by time, including state, action, reward and new state
    def plot_transition(self, t, bs=0):
        # print selected action
        action = self.buffer["actions"][bs, t, 0, 0]
        avail_actions = th.where(self.buffer["avail_actions"][bs, t, 0, :])[0].cpu().tolist()
        reward = self.buffer["reward"][bs, t, 0].item()
        print(f'Action: {action}\t Available Actions: {avail_actions}\t Reward: {reward}')
        
        state = self.buffer["obs"][bs, t, 0, :].reshape(self.height, self.width, -1)
        state = (state[:, :, 1] * (1 - 3*state[:, :, 0]) + state[:, :, 2] + 8*(state[:, :, 3] == -1)).cpu().tolist()
        state = [[self.state_repr[e] for e in row] for row in state]

        new_state = self.buffer["obs"][bs, t+1, 0, :].reshape(self.height, self.width, -1)
        new_state = (new_state[:, :, 1] * (1 - 3*new_state[:, :, 0]) + new_state[:, :, 2] + 8*(new_state[:, :, 3] == -1)).cpu().tolist()
        new_state = [[self.state_repr[e] for e in row] for row in new_state]

        for i in range(len(state)):
            print(str(state[i]).replace("'","") + '\t' + str(new_state[i]).replace("'",""))
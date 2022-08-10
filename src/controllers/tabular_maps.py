import numpy as np
import torch as th
from os import path as pt
from tabulate import tabulate
from .basic_controller import BasicMAC
from action_model import REGISTRY as model_REGISTRY
from modules.agents import REGISTRY as agent_REGISTRY
th.set_printoptions(precision=2)

class TabularMAPS(BasicMAC):
    def __init__(self, scheme, groups, args):
        super().__init__(scheme, groups, args)
        self.n_actions = args.n_actions
        
        # MAPS properties & action model initalization
        self.random_ordering = getattr(args, "random_ordering", True)
        self.action_model = model_REGISTRY[args.env](scheme, args)
        self.cliques = np.empty(0) # count number of single-steps in the previous iteration

        self.qtable = agent_REGISTRY[args.agent](None, args)

    ### This function overrides MAC's original function because MAPS selects actions sequentially and select actions cocurrently ###
    def select_actions(self, ep_batch, t_ep, t_env, bs=..., test_mode=False):
        # Update state of the action model based on the results
        state_data = self.action_model.update_state(ep_batch, t_ep, test_mode)

        #$ DEBUG: Reset logger - record all q-values during an episode
        if t_ep == 0:
            # if hasattr(self, 'logger'):
            #     self.values_seq()
            self.logger = th.zeros((0, self.n_actions))

            self.test = test_mode #$
            self.cliques = np.empty(0) #! This is not debug
            self.batch = self.action_model.batch

        # Detect interactions between agent, for selecting a joint action in these interactions
        self.cliques = self.action_model.detect_interaction(state_data)
        if self.random_ordering:
            self.cliques = np.random.permutation(self.cliques)

        # Array to store the chosen actions
        chosen_actions = th.zeros((1, self.n_agents), dtype=th.int)

        # MAPS core - runs the agents sequentially based on the chosen order
        for i in self.cliques:
            # get (pseudo-)observation batch from action model
            obs, data = self.action_model.get_obs_agent(i)
            probs = data["probs"].view(-1, 1)
            
            # calculate values of the current (pseudo-)state batch and select action
            avail_actions = self.action_model.get_avail_actions(data, i, ep_batch["avail_actions"][:, t_ep, i]).unsqueeze(dim=1)

            # Calculate action based on pseudo-state
            inputs = self._build_inputs(obs.unsqueeze(dim=1), self.action_model.batch, self.action_model.t)
            values = self.select_agent_action(inputs)
            values = th.unsqueeze((values * probs.view(1, -1, 1)).sum(dim=1), dim=1)

            chosen_actions[0, i] = self.action_selector.select_action(values[bs], avail_actions, t_env, test_mode=test_mode)

            # simulate action in the environment
            self.action_model.step(i, chosen_actions, obs, data["hidden"], avail_actions)

            #$ DEBUG: Log q-values in the logger
            self.logger = th.cat((self.logger, th.squeeze(values, axis=1)), axis=0)

        # Online learning for the previous timestep
        if not test_mode and t_ep > 0:
            max_i = th.where(self.action_model.batch["terminated"])[1][0].item() if self.action_model.terminated else 1e7
            rng = slice(self.n_agents*(t_ep-1), min(self.n_agents*t_ep, max_i)+1)

            self.agent.update_qvalues(
                self.batch["obs"][:, rng , 0],
                self.batch["actions"][:, rng, 0, 0],
                self.batch["avail_actions"][:, rng, 0],
                self.batch["reward"][:, rng, 0]
            )
                #!!!! CHECK THIS !!!!#

        return chosen_actions

    def _build_inputs(self, obs, batch, t):
        return obs.reshape(obs.shape[0], -1)

    # Calculate q-values based on observation
    def select_agent_action(self, obs):
        agent_outs = self.agent.forward(obs)
        return agent_outs.view(1, obs.shape[0], -1)
   
    def init_hidden(self, batch_size):
        pass

   # Used for training and propagating the hidden state in stochastic environment, not for action selection
    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch["obs"][:, t], ep_batch, t).view(ep_batch.batch_size, 1, -1)
        agent_outs = self.agent.forward(agent_inputs)
        
        return agent_outs.view(ep_batch.batch_size, 1, -1)

    #$ DEBUG: Plot the sequence of q-values for the whole episode
    def values_seq(self):
        bs = self.action_model.buffer.buffer_index-1
        values_data =[]
        headers = ['Time', 'Agent', 'Q-Values', 'Max Q-value', 'Avail Actions', 'Action', 'Reward', 'Target']

        for t in range(min(self.logger.shape[0], self.action_model.buffer['reward'].shape[1])):
            values = self.logger[t, :]
            reward = self.action_model.buffer['reward'][bs, t, 0]
            action = self.action_model.buffer["actions"][bs, t, 0, 0]
            avail_actions = th.where(self.action_model.buffer["avail_actions"][bs, t, 0, :])[0].cpu().tolist()
            
            avail_values  = values.clone()
            avail_values[self.action_model.buffer["avail_actions"][bs, t, 0, :] == 0] = -1e7
            
            if t+1 < min(self.logger.shape[0], self.action_model.buffer['reward'].shape[1]) - 1:
                n_values = self.logger[t+1, :].clone() - 1e7 * (1 - self.action_model.buffer["avail_actions"][bs, t, 0, :])
                target = th.max(n_values) * self.args.gamma + reward
            else:
                target = reward

            if len(avail_actions) > 0:
                values_data.append([t, t%self.n_agents, values.data, th.max(avail_values), avail_actions, action, reward, target])
        print(tabulate(values_data, headers=headers, numalign='center', tablefmt="github"))

    #$ DEBUG: Plot the q-values of a transition
    # prev flag - debug the previous, completed episodes, hence take data from previous episode
    def plot_transition(self, t):
        values = self.logger[t, :]
        print(f'Q-values: {values.data}\t Best action: {th.argmax(values)}')
        self.action_model.plot_transition(t, bs=self.action_model.buffer.buffer_index-1)

    #$ DEBUG: Find the number of episode in previous run
    def last_transition(self):
        bs=self.action_model.buffer.buffer_index-1
        return th.sum(self.action_model.buffer['filled'][bs, :, 0])

    # Irrelevant inhereted functions
    def load_state(self, other_mac):
        pass

    def cuda(self):
        pass

    def save_models(self, path):
        path = pt.dirname(path)
        self.agent.save_model(path)

    def load_models(self, path):
        path = pt.dirname(path)
        self.agent.load_model(path)
import numpy as np
import torch as th
from tabulate import tabulate
from .basic_controller import BasicMAC
from action_model import REGISTRY as model_REGISTRY

class PSeqMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super().__init__(scheme, groups, args)
        self.n_actions = args.n_actions
        
        # PSeq properties & action model initalization
        self.random_ordering = getattr(args, "random_ordering", True)
        self.action_model = model_REGISTRY[args.env](scheme, args)
        self.cliques = np.empty(0) # count number of single-steps in the previous iteration
        # ! CUDA - how to move calcs to GPU
            

    ### This function overrides MAC's original function because PSeq selects actions sequentially and select actions cocurrently ###
    def select_actions(self, ep_batch, t_ep, t_env, bs=..., test_mode=False):
        # Update state of the action model based on the results
        self.action_model.update_state(ep_batch["state"][:, t_ep], t_ep, test_mode)

        # Preservation of hidden state in stochastic environment
        if self.action_model.stochastic_env:
            if t_ep == 0: # init hidden state
                self.known_hidden_state = self.hidden_states.clone()
            else: # calculate new hidden state based on the real outcomes
                self._propagate_hidden(steps=len(self.cliques))   

        # Detect interactions between agent, for selecting a joint action in these interactions
        self.cliques = self.action_model.detect_interaction()
        if self.random_ordering:
            self.cliques = np.random.permutation(self.cliques)

        #$ DEBUG: Reset logger - record all q-values during an episode
        if t_ep == 0:
            self.logger = th.zeros((0, self.n_actions)).detach()
            
        # Array to store the chosen actions
        chosen_actions = th.zeros((1, self.n_agents), dtype=th.int)
        # PSeq core - runs the agents sequentially based on the chosen order
        for i in self.cliques:
            # get (pseudo-)observation batch from action model
            obs = th.unsqueeze(self.action_model.get_obs_agent(i), dim=0)
            
            # calculate values of the current (pseudo-)state batch and select action
            avail_actions = self.action_model.get_avail_actions(i, ep_batch["avail_actions"][:, t_ep, i]).unsqueeze(dim=1)

            # calculate action based on pseudo-state
            values = self.select_agent_action(self._build_inputs(obs, self.action_model.batch, self.action_model.t), i)
            chosen_actions[0, i] = self.action_selector.select_action(values[bs], avail_actions, t_env, test_mode=test_mode)

            # simulate action in the environment
            self.action_model.step(i, chosen_actions, obs, avail_actions)

            #$ DEBUG: Log q-values in the logger
            self.logger = th.cat((self.logger, th.squeeze(values, axis=1)), axis=0)

        return chosen_actions

    def _build_inputs(self, obs, batch, t):
        bs = batch.batch_size
        inputs = [obs]

        if self.args.obs_last_action:
            last_action = th.zeros_like(batch["actions_onehot"][:, t]) if t == 0 else batch["actions_onehot"][:, t-1]
            inputs.append(last_action)
        return th.cat([x.reshape(obs.shape[0], -1) for x in inputs], dim=1) #! bs?????

    #  calculate q-values based on observation
    def select_agent_action(self, obs, i):
        agent_outs, self.hidden_states = self.agent(obs, self.hidden_states)
        return agent_outs.view(1, 1, -1)
   
    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, 1, -1)

    # Used only for training, not for action selection
    def forward(self, ep_batch, t, test_mode=False):
        # agent_inputs = self._build_inputs(ep_batch["obs"][:, t], self.action_model.buffer, t).view(ep_batch.batch_size, -1)
        agent_inputs = self._build_inputs(ep_batch["obs"][:, t], ep_batch, t).view(ep_batch.batch_size, -1)
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        return agent_outs.view(ep_batch.batch_size, 1, -1)

    # Update the hidden state based on the real outcomes (rather than estimated outcomes as calculated during the sequential run)
    def _propagate_hidden(self, steps):
        self.hidden_states = self.known_hidden_state
        for s in range(steps, 0, -1):
            self.forward(self.action_model.batch, t=self.action_model.t-s)
        self.known_hidden_state = self.hidden_states.clone()




    #$ DEBUG: Plot the sequence of q-values for the whole episode
    def values_seq(self):
        bs = self.action_model.buffer.buffer_index-1
        values_data =[]
        headers = ['Time', 'Agent', 'Q-Values', 'Max Q-value', 'Avail Actions', 'Action', 'Reward']

        for t in range(min(self.logger.shape[0], self.action_model.buffer['reward'].shape[1])):
            values = self.logger[t, :]
            reward = self.action_model.buffer['reward'][bs, t, 0]
            action = self.action_model.buffer["actions"][bs, t, 0, 0]
            avail_actions = th.where(self.action_model.buffer["avail_actions"][bs, t, 0, :])[0].cpu().tolist()
            if len(avail_actions) > 0:
                values_data.append([t, t%self.n_agents, values.data, th.max(values[avail_actions]), avail_actions, action, reward])
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
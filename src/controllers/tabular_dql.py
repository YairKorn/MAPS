import numpy as np
import torch as th
from tabulate import tabulate
from .basic_controller import BasicMAC
from action_model import REGISTRY as model_REGISTRY
# from test.agent_maps_test import MAPSTest
th.set_printoptions(precision=2)

class TabularDQL(BasicMAC):
    def __init__(self, scheme, groups, args):
        super().__init__(scheme, groups, args)
        self.n_actions = args.n_actions
        
        # MAPS properties & action model initalization
        self.random_ordering = getattr(args, "random_ordering", True)
        self.action_model = model_REGISTRY[args.env](scheme, args)
        self.cliques = np.empty(0) # count number of single-steps in the previous iteration

    ### This function overrides MAC's original function because MAPS selects actions sequentially and select actions cocurrently ###
    def select_actions(self, ep_batch, t_ep, t_env, bs=..., test_mode=False):
        # Update state of the action model based on the results
        state_data = self.action_model.update_state(ep_batch, t_ep, test_mode)

        #$ DEBUG: Reset logger - record all q-values during an episode
        if t_ep == 0:
            self.logger = th.zeros((0, self.n_actions))
            self.test = test_mode #$
            self.cliques = np.empty(0) #! This is not debug

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
            
            # calculate values of the current (pseudo-)state batch and select action
            avail_actions = ep_batch["avail_actions"][:, t_ep, i, :].unsqueeze(1)
            agent_outputs = self.forward(ep_batch, t_ep, i, test_mode=test_mode)

            chosen_actions[0, i] = self.action_selector.select_action(agent_outputs[bs], avail_actions, t_env, test_mode=test_mode)
            self.action_model.step(i, chosen_actions, obs, data["hidden"], avail_actions)

        return chosen_actions

    def forward(self, ep_batch, t, i=..., test_mode=False):
        n_agents = self.n_agents if ep_batch["obs"].shape[0] > 1 else 1
        agent_inputs = self._build_inputs(ep_batch["obs"][:, t, i]).view(ep_batch.batch_size, n_agents, -1)
        agent_outs = self.agent.forward(agent_inputs)
        
        return agent_outs.view(ep_batch.batch_size, n_agents, -1)

    def _build_inputs(self, obs, agents=1):
        return obs.reshape(obs.shape[0], agents, -1)

    def init_hidden(self, batch_size):
        pass


    # Irrelevant inhereted functions
    def load_state(self, other_mac):
        pass

    def cuda(self):
        pass

    def save_models(self, path):
        pass

    def load_models(self, path):
        pass
import numpy as np
import torch as th
from .basic_controller import BasicMAC
from action_model import REGISTRY as model_REGISTRY

class PSeqMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super().__init__(scheme, groups, args)
        self.n_actions = args.n_actions
        
        # PSeq properties & action model initalization
        self.random_ordering = getattr(args, "random_ordering", True)
        self.action_model = model_REGISTRY[args.env](scheme, args)

        # ! CUDA - how to move calcs to GPU
        # ! parameters, save & load models - check if need to be changed

    ### This function overrides MAC's original function because PSeq selects actions sequentially and select actions cocurrently ###
    def select_actions(self, ep_batch, t_ep, t_env, bs=..., test_mode=False):
        # Update state of the action model based on the results
        # return a list of "augmanted agents" - agents that in interaction and need to select a joint action
        interaction_cg = self.action_model.update_state(ep_batch["state"][:, t_ep], t_ep, test_mode)
        if self.random_ordering:
            interaction_cg = np.random.permutation(interaction_cg)

        # if self.args.obs_last_action:
        #     self.step_inputs.append(th.zeros_like(ep_batch["actions_onehot"][:, t_ep]) if t_ep == 0 else ep_batch["actions_onehot"][:, t_ep-1])

        # Array to store the chosen actions
        chosen_actions = th.zeros((1, self.n_agents), dtype=th.int)
        
        # PSeq core - runs the agents sequentially based on the chosen order
        for i in interaction_cg:
            # get (pseudo-)observation batch from action model
            obs = th.unsqueeze(self.action_model.get_obs_agent(i), dim=0)
            
            # calculate values of the current (pseudo-)state batch and select action
            avail_actions = self.action_model.get_avail_actions(i, ep_batch["avail_actions"][:, t_ep, i]).unsqueeze(dim=1)

            # calculate action based on pseudo-state
            values = self.select_agent_action(self._build_inputs(obs, self.action_model.batch, self.action_model.t), i)
            chosen_actions[0, i] = self.action_selector.select_action(values[bs], avail_actions, t_env, test_mode=test_mode)

            # simulate action in the environment
            self.action_model.step(i, chosen_actions, obs, avail_actions)

        return chosen_actions

    def _build_inputs(self, obs, batch, t):
        bs = batch.batch_size
        inputs = [obs]

        if self.args.obs_last_action:
            last_action = th.zeros_like(batch["actions_onehot"][:, t]) if t == 0 else \
                batch["actions_onehot"][:, t-1]
            inputs.append(last_action)
        return th.cat([x.reshape(obs.shape[0], -1) for x in inputs], dim=1)

    #  calculate q-values based on observation
    def select_agent_action(self, obs, i):
        agent_outs, self.hidden_states = self.agent(obs, self.hidden_states)
        return agent_outs.view(1, 1, -1)
   
    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, 1, -1)

    def forward(self, ep_batch, t, test_mode=False):
        #!!! I need to correct this!!!
        agent_inputs = self._build_inputs(ep_batch["obs"][:, t], self.action_model.buffer, t).view(ep_batch.batch_size, -1)
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        return agent_outs.view(ep_batch.batch_size, 1, -1)

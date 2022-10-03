import numpy as np
import torch as th
from os import path as pt
from .basic_controller import BasicMAC
from modules.agents import REGISTRY as agent_REGISTRY
th.set_printoptions(precision=2)

class TabularDQL(BasicMAC):
    def __init__(self, scheme, groups, args):
        super().__init__(scheme, groups, args)
        self.n_actions = args.n_actions
        self.qtable = agent_REGISTRY[args.agent](None, args)

    ### This function overrides MAC's original function because MAPS selects actions sequentially and select actions cocurrently ###
    def select_actions(self, ep_batch, t_ep, t_env, bs=..., test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)

        # Online learning for the previous timestep
        if not test_mode and t_ep > 0:
            rng = slice(t_ep-1, t_ep+1)

            self.agent.update_qvalues(
                ep_batch["obs"][:, rng , :],
                ep_batch["actions"][:, rng, :, 0],
                ep_batch["avail_actions"][:, rng, :],
                ep_batch["reward"][:, rng, :].expand(1, 2, self.n_agents) / self.n_agents
            )
            #! NOTE THAT REWARD IS NOT DECOMPOSED, BUT DIVIDED EQUALLEY AMONG AGENTS

        return chosen_actions

    def _build_inputs(self, batch, t):
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs


    # Calculate q-values based on observation
    def select_agent_action(self, obs):
        agent_outs = self.agent.forward(obs)
        return agent_outs.view(1, obs.shape[0], -1)
   
    def init_hidden(self, batch_size):
        pass

   # Used for training and propagating the hidden state in stochastic environment, not for action selection
    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        agent_outs = self.agent.forward(agent_inputs)

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

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
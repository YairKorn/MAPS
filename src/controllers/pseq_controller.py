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

        # TODO: initialize agents using args - DQN / DDQN / RNN etc.
        # ! CUDA - how to move calcs to GPU
        # ! parameters, save & load models - check if need to be changed

    ### This function overrides MAC's original function because PSeq selects actions sequentially and select_actions selects actions cocurrently ###
    def select_actions(self, ep_batch, t_ep, t_env, bs=..., test_mode=False):
        # Update state of the action model based on the results
        # return a list of "augmanted agents" - agents that in interaction and need to select a joint action
        interaction_cg = self.action_model.update_state(ep_batch["state"][:, t_ep])

        # if self.args.obs_last_action:
        #     self.step_inputs.append(th.zeros_like(ep_batch["actions_onehot"][:, t_ep]) if t_ep == 0 else ep_batch["actions_onehot"][:, t_ep-1])

        # Array to store the chosen actions
        chosen_actions = th.zeros((1, self.n_agents))

        # choose execution sequence
        if self.random_ordering:
            interaction_cg = np.random.permutation(interaction_cg)
        
        # PSeq core - runs the agents sequentially based on the chosen order
        for i in interaction_cg:
            # get (pseudo-)observation batch from action model
            obs = self.action_model.get_obs_agent(i)
            
            # calculate values of the current (pseudo-)state batch and select action
            avail_actions = ep_batch["avail_actions"][:, t_ep, i].unsqueeze(dim=1)

            # calculate action based on pseudo-state
            values = self.select_agent_action(obs, i)
            chosen_actions[0, i] = self.action_selector.select_action(values[bs], avail_actions, t_env, test_mode=test_mode)

            # simulate action in the environment
            self.action_model.step(i, chosen_actions)

        return chosen_actions

    # get the observation from action model and calculate based on this observation
    def select_agent_action(self, obs, i):

        agent_inputs, obs_probs = self._build_inputs(i)
        agent_outs, self.hidden_states = self.agent(obs, self.hidden_states)
        
        # augment q-values/probabiliteis over possible states
        agent_outs = th.sum(obs_probs * agent_outs, dim=0)
        #### TODO build the agent-core network (DDQN)

        return agent_outs.view(ep_batch.batch_size, len(i) if type(i)==list else 1, -1)
   
    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    # override BasicMAC _build_inputs, and get observation from action model #! after building obs, may need to change to observation.shape[1]
    def _build_inputs(self, i):
        observation, obs_prob = th.rand((self.n_agents, 400), device=th.device('cuda')), th.rand((self.n_agents, 1), device=th.device('cuda')) #! self.action_model.get_obs_agent(i)
        inputs = [observation] + [x[:, i].repeat((observation.shape[0], 1)) for x in self.step_inputs]
        return th.cat([x.reshape(observation.shape[0], -1) for x in inputs], dim=1), obs_prob

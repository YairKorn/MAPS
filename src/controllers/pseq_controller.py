import numpy as np
from .basic_controller import BasicMAC
from src.action_model import REGISTRY as model_REGISTRY

class PSeqMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super().__init__(scheme, groups, args)
        self.n_actions = args.n_actions
        
        # PSeq properties & action model initalization
        self.random_ordering = getattr(args, "random_ordering", True)
        self.action_model = model_REGISTRY[args.environment](args)

        # TODO: initialize agents using args - DQN / DDQN / RNN etc.
        # ! CUDA - how to move calcs to GPU
        # ! parameters, save & load models - check if need to be changed

    ### This function overrides MAC's original function because PSeq selects actions sequentially and select_actions selects actions cincurrently ###
    def select_actions(self, ep_batch, t_ep, t_env, bs=..., test_mode=False):
        # Update state of the action model based on the results
        # return a list of "augmanted agents" - agents that should considered as one agent
        augmanted_agents = self.action_model.update(ep_batch["state"][:, t_ep])

        # Array to hold the chosen actions
        chosen_actions = np.empty(shape=(1, self.n_agents))

        # choose execution sequence
        execute_order = np.random.permutation(augmanted_agents) if self.random_ordering else np.arange(augmanted_agents)
        
        # PSeq core - run the agents sequentially based on the chosen order
        for i in execute_order:
            # get (pseudo-)state batch from action model
            state_batch = self.action_model.get_obs_agent(i)
            
            # calculate values of the current (pseudo-)state batch and select action
            # * this section can be extended for policy-based or actor-critic algorithms
            avail_actions = ep_batch["avail_actions"][i, t_ep]
            values = self.select_agent_action(state_batch) # check that availd[i, t_ep] is the right argument
            chosen_actions[0, i] = self.action_selector.select_action(values[bs], avail_actions, t_env, test_mode=test_mode)

            # simulate action in the environment
            self.action_model.step(i, chosen_actions)

        return chosen_actions

    
    def select_agent_action(self, state_batch):
        agent_inputs = self._build_inputs(ep_batch, t) # TODO fill
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
        
        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)
        # TODO make sure that the output is correct (q values?)
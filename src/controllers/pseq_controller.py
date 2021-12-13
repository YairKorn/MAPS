import numpy as np
from .basic_controller import BasicMAC
from src.action_model import REGISTRY as model_REGISTRY

class PSeqMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super().__init__(scheme, groups, args)
        self.n_actions = args.n_actions
        
        # PSeq properties
        self.random_ordering = getattr(args, "random_ordering", True)

        # action model initalization
        self.action_model = model_REGISTRY[args.environment](args)


    ### This function overrides MAC's original function because PSeq consider the selected actions ###
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
            state_batch = self.action_model.get_obs_agent(i)
            chosen_actions[0, i] = self.select_agent_action(state_batch, ep_batch["avail_actions"][i, t_ep]) # check that availd[i, t_ep] is the right argument
            # TODO: consider use of "action_selector" and let select_agent_action to return q-values or probabilities
            self.action_model.step(i, chosen_actions)

        return chosen_actions

        # TODO: complete this function

    
    def select_agent_action(self, state_batch, avail_actions=None):
        # A list
        pass
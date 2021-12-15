import sys
import numpy as np
from collections import namedtuple

BufferEntry = namedtuple('BufferEntry', ['id', 'prev_state', 'action', 'new_state'])

""" Basic action model class - action models inherit from it """
class ActionModel():
    def __init__(self, args) -> None:
        # stochastic environment properties
        self.stochastic_env  = getattr(args, "stochastic_environment", False)
        # ... if monte_carlo is off, set MCTS maximal batch size to be maxsize, such that in practice MCTS is not used
        self.mcts_batch_size = getattr(args, "mcts_batch_size", 32) if getattr(args, "apply_monte_carlo", True) else sys.maxsize

        # general
        self.actions = np.ones(shape=(1, args.n_actions)) # the probabilities for different outcomes of actions; self.actions[action].shape = (-1, 1)
        self.buffer = np.empty(shape=0) # np.full(shape=getattr(args, "n_agents"), fill_value=np.nan)

    """ When new perception is percepted, update the real state """
    def update_state(self, state):
        self.state, self.state_prob = state, np.ones(1)

    """ Update the state based on an action """
    def step(self, action):
        self.state, self.state_prob = self._MCTS(action) if self.stochastic_env else self._apply_action_on_state(self.state[0], action), 1.0

    """ For stochastic environents (if "apply_monte_carlo" is True), use monte-carlo to select possible outcomes """
    def _MCTS(self, action):
        different_states = np.shape(self.state_prob)[0]
        # calculate the probabilites of all possible outcomes, based on the assumed state and the action possible outcomes
        possible_outcomes = np.tensordot(self.state_prob, self.actions[action], 0).reshape(-1)

        # use monte-carlo to select randomly the states to consider
        mc_outcomes = np.arange(possible_outcomes.shape[0]) if possible_outcomes.shape[0] > self.mcts_batch_size else \
            np.random.choice(possible_outcomes.shape[0], size=self.mcts_batch_size, replace=False, p=possible_outcomes)        

        # return a new batch of states & their new normalized probabilities
        return [self._apply_action_on_state(self.state[int(i / different_states)], action, i % different_states) for i in mc_outcomes], \
            possible_outcomes[mc_outcomes] / np.sum(possible_outcomes[mc_outcomes])
    
    def _apply_action_on_state(self, state, action, result=0):
        raise NotImplementedError


# class StateUnit():
#     def __init__(self, state, prob) -> None:
#         self.state = state
#         self.prob  = prob
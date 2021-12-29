import sys
import numpy as np
import torch as th
from copy import deepcopy
from collections import namedtuple, deque
from copy import deepcopy

BufferEntry = namedtuple('BufferEntry', ['id', 'prev_state', 'action', 'new_state', 'reward'])

""" Basic action model class - action models inherit from it """
class ActionModel():
    def __init__(self, args) -> None:
        self.n_agents = args.n_agents
        self.device = "cuda" if args.use_cuda else "cpu"

        # stochastic environment properties
        self.stochastic_env = getattr(args, "stochastic_environment", False)
        
        # ... if monte_carlo is off, set MCTS maximal batch size to be maxsize, such that in practice MCTS is not used
        mc_batch_size = getattr(args, "mcts_batch_size", 32)
        self.mcts_batch_size = mc_batch_size if getattr(args, "apply_monte_carlo", True) else sys.maxsize

        # partial observability
        self.full_observable = getattr(args, "observe_state", False) #! make sure that it works

        # general
        self.actions = np.ones(shape=(1, args.n_actions)) # the probabilities for different outcomes of actions; self.actions[action].shape = (-1, 1)
        self.buffer = deque()                             # store episodes for training

        # Buffer to store episodes for the training process
        self.learning_buffer = th.Tensor()

    """ When new perception is percepted, update the real state """
    def update_state(self, state):
        self.state, self.state_prob = state, np.ones(1)

    """ Update the state based on an action """
    def step(self, agent_id, action):
        prev_state = deepcopy(self.state)
        self.state, self.state_prob = self._MCTS(action) if self.stochastic_env else self._apply_action_on_state(self.state[0], action), 1.0
        
        # Temporary buffer until new perception from the environment is gathered
        buffer_entry = BufferEntry(agent_id, prev_state, action, deepcopy(self.state), 0)
        self.buffer.append(buffer_entry)

    """ For stochastic environents, use monte-carlo to select possible outcomes """
    def _MCTS(self, action):
        # How many states are possible before the action of the agent
        no_states = np.shape(self.state_prob)[0]
        
        # Calculate the probabilites of all possible outcomes, based on the assumed state and the action possible outcomes
        pos_outcomes = np.tensordot(self.state_prob, self.actions[action], 0).reshape(-1)
        no_outcomes = pos_outcomes.shape[0]
        
        # Use MC to select randomly the states to consider
        mc_outcomes = np.arange(no_outcomes) if no_outcomes > self.mcts_batch_size else \
            np.random.choice(no_outcomes, size=self.mcts_batch_size, replace=False, p=pos_outcomes)        

        # return a new batch of states & their new normalized probabilities
        #! doesn't work
        return [self._apply_action_on_state(self.state[int(i / no_states)], action, i % no_states) for i in mc_outcomes], \
            pos_outcomes[mc_outcomes] / np.sum(pos_outcomes[mc_outcomes])
    

    # Use the general state to create a partial-observability observation for the agents
    def _build_observation(self, i):
        if self.full_observable:
            return deepcopy(self.state)

        #! build


    def _apply_action_on_state(self, state, action, result=0):
        raise NotImplementedError


    def _detect_interaction(self, state):
        return np.arange(self.n_agents)


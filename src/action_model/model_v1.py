import numpy as np
import torch as th
from copy import deepcopy
from collections import namedtuple, deque
from components.episode_buffer import ReplayBuffer

BufferEntry = namedtuple('BufferEntry', ['obs', 'action', 'avail_actions', 'new_obs', 'reward'])

""" Basic action model class - action models inherit from it """
class ActionModel():
    def __init__(self, scheme, args) -> None:
        self.n_agents = args.n_agents
        self.device = "cuda" if args.use_cuda else "cpu"

        # Buffer for storing episodes until the results of the action are revealed (by the next perception)
        #! need to implement in stochastic env (v2/v3)
        self.state_buffer = deque()

        # Buffer for sequential single-agent samples (rather than n-agents samples) #$ (Fits to dynamic CG)
        model_scheme = {
            "obs": scheme["obs"],
            "actions": scheme["actions"],
            "avail_actions": scheme["avail_actions"],
            "new_obs": scheme["obs"],
            "reward": scheme["reward"],
            "terminated": scheme["terminated"],
        }
        groups = {'agents': self.n_agents}
        self.episode_limit = args.env_args['episode_limit']
        self.buffer = ReplayBuffer(model_scheme, groups, args.buffer_size, self.n_agents * self.episode_limit + 1, device=self.device)
        pass

    """ When new perception is percepted, update the real state """
    def update_state(self, state):
        self.state = state
        return np.arange(self.n_agents)

    """ Update the state based on an action """
    def step(self, agent_id, action):
        prev_state = deepcopy(self.state)
        self.state = self._apply_action_on_state(self.state, action)
        
        #! Temporary buffer until new perception from the environment is gathered
        # buffer_entry = BufferEntry(agent_id, prev_state, action, deepcopy(self.state), 0)
        # self.buffer.append(buffer_entry)


    # Use the general state to create a partial-observability observation for the agents
    def _build_observation(self, i):
        if self.full_observable:
            return deepcopy(self.state)

        #! build


    def _apply_action_on_state(self, state, action, result=0):
        raise NotImplementedError


    def get_obs_agent(self, agent_id):
        pass


    """ This function build a dynamic CG using pre-defined (problem specific) model """
    def _detect_interaction(self, state):
        return np.arange(self.n_agents)


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

        # Buffers for store learning episodes (because the default buffer is not fit)
        model_scheme = {
            "obs": scheme["obs"],
            "actions": scheme["actions"],
            "avail_actions": scheme["avail_actions"],
            # "new_obs": scheme["obs"],
            "reward": scheme["reward"],
            "terminated": scheme["terminated"],
        }
        groups = {'agents': self.n_agents}
        self.episode_limit = args.env_args['episode_limit']

        # Buffer for storing episodes until the results of the action are revealed (by the next perception)
        #! need to implement in stochastic env (v2/v3)
        #* self.temp_buffer = ReplayBuffer(model_scheme, groups, args.buffer_size, self.n_agents, device=self.device)

        # Buffer for sequential single-agent samples (rather than n-agents samples) #$ (Fits to dynamic CG)
        self.buffer = ReplayBuffer(model_scheme, groups, args.buffer_size, self.n_agents * self.episode_limit + 1, device=self.device)


    """ When new perception is percepted, update the real state """
    def update_state(self, state):
        self.state = state
        return np.arange(self.n_agents)

    """ Update the state based on an action """
    def step(self, agent_id, action, obs, avail_actions):
        self.state, reward, terminated = self._apply_action_on_state(self.state, action)
        
        action_data = {
            "obs": obs,
            "avail_actions": avail_actions,
            "actions": action * self._one_hot((1, self.n_agents), (agent_id,)),
            "reward": reward,
            "terminated": terminated
        }
        self.buffer.update(action_data, ts=t)


        
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

    @staticmethod
    def _one_hot(shape, one_hot):
        arr = np.zeros(shape)
        arr[one_hot] = 1
        return arr
import numpy as np
from components.episode_buffer import ReplayBuffer
from utils.dict2namedtuple import convert

""" Basic action model class - action models inherit from it """
class ActionModel():
    def __init__(self, scheme, args) -> None:
        self.n_agents = args.n_agents
        self.device = "cuda" if args.use_cuda else "cpu"
        self.t = 0

        # Unpack arguments from sacred
        if isinstance(args, dict):
            args = convert(args)
        self.args = args

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

        # Buffer for sequential single-agent samples (rather than n-agents samples) #* (Fits to dynamic CG)
        self.buffer = ReplayBuffer(model_scheme, groups, args.buffer_size, self.n_agents * self.episode_limit + 1, device=self.device)


    """ When new perception is percepted, update the real state """
    def update_state(self, state):
        self.state = state
        return self._detect_interaction()


    """ Update the state based on an action """
    def step(self, agent_id, actions, obs, avail_actions):
        reward, terminated = self._apply_action_on_state(agent_id, actions[0, agent_id])
        
        transition_data = {
            "obs": obs,
            "avail_actions": avail_actions,
            "actions": actions[agent_id] * self._one_hot((1, self.n_agents), (agent_id,)),
            "reward": reward,
            "terminated": terminated
        }
        
        self.buffer.update(transition_data, ts=self.t)
        self.t += 1

        if terminated:
            self.buffer.update({
                "obs": self.state
            }, ts=self.t)


    """ Use the general state to create a partial-observability observation for the agents """
    def get_obs_agent(self, agent_id):
        raise NotImplementedError

    """ Simulate the result of action in the environment """
    def _apply_action_on_state(self, agent_id, action, result=0):
        raise NotImplementedError

    """ This function build a dynamic CG using pre-defined (problem specific) model """
    def _detect_interaction(self):
        return np.arange(self.n_agents) # by default, no interaction

    @staticmethod
    def _one_hot(shape, one_hot):
        arr = np.zeros(shape)
        arr[one_hot] = 1
        return arr
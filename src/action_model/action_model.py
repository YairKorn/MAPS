import numpy as np
import torch as th
from functools import partial
from components.episode_buffer import EpisodeBatch, ReplayBuffer
from components.mcts_buffer import MCTSBuffer
from utils.dict2namedtuple import convert

""" Basic action model class - action models inherit from it """
class ActionModel():
    def __init__(self, scheme, args) -> None:
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.buffer_size = args.buffer_size
        self.device = "cuda" if (args.use_cuda and not args.internal_buffer_cpu) else "cpu"

        # Mechanisms for stochastic environment
        self.stochastic_env = True              # TRUE by default - determinstic action models should override this attribute
        self.action_order = []                  # order of the agents that performed actions, used to back-updating the observations
        self.MCTS_sampling = args.MCTS_sampling # size of sample every action selection
        self.mcts_buffer = MCTSBuffer(self._get_data_shape(scheme, args), args.MCTS_buffer_size, device=self.device)
        print("### MCTS mode active ###")

        # Episodes management (no need in general, but for compatability with PyMARL)
        self.env_t = 0              # real time of the environment
        self.t = 0                  # time in the episode
        self.test_mode = False      # mode, for preventing training on test episodes
        self.terminated = False     # used to prevent saving steps after the episode was ended

        # Unpack arguments from sacred
        if getattr(args, "env_args", None):
            args = args.env_args
        if isinstance(args, dict):
            args = convert(args)
        self.args = args

        # Buffers for store learning episodes (because the default buffer is not fit)
        model_scheme = {
            "obs": scheme["obs"],
            "actions": scheme["actions"],
            "avail_actions": scheme["avail_actions"],
            "actions_onehot": scheme["avail_actions"],
            "reward": scheme["reward"],
            "terminated": scheme["terminated"],
        }
        groups = {'agents': 1} # treat each agent to act in a specific timestep
        self.episode_limit = args.episode_limit

        # Buffer for sequential single-agent samples (rather than n-agents samples) #* (Fits to dynamic CG)
        self.new_batch = partial(EpisodeBatch, model_scheme, groups, 1, self.episode_limit * self.n_agents + 1, preprocess=None, device=self.device)
        self.buffer = ReplayBuffer(model_scheme, groups, self.buffer_size, self.episode_limit * self.n_agents + 1, device=self.device)

    """ When new perception is percepted, update the real state """
    def update_state(self, state, t_ep, test_mode):
        if t_ep == 0:  # when the env time resets, a new episode has begun
            self.batch = self.new_batch()       # new batch for storing steps
            self.test_mode = test_mode          # does episode is for test or training
            self.t = 0                          # reset internal time
            self.terminated = False

        self.env_t = t_ep                       # update environment real time
        data = self._update_env_state(state)    # update state (env-specific method) - used for updating stochastic results and extract state features
        self.mcts_buffer.reset(data)

        # In case of stochastic environment, updating the previous observations based on the new observations...
        if self.stochastic_env and self.t > 0:
            # ... iterate over agents to update the observation
            for s in range(1, len(self.action_order)):
                self._back_update(self.batch, data, self.t-len(self.action_order)+s, s) # tensors share physical memory
        self.action_order = [] # reset order of actions


    """ Update the state based on an action """
    def step(self, agent_id, actions, obs, avail_actions):
        self.action_order.append(agent_id)

        # PSeq assumpsion: transition function can be decomposed into the product of transition functions of the cliques
        # therefore, sample and e(xample) state from MCTS buffer to calculate possible result
        data = self.mcts_buffer.sample(sample_size=1)
        p_result = self._action_results(data, agent_id, actions[0, agent_id])

        results, probs = self.mcts_buffer.mcts_step(p_result)
        dpack = [{k:v[i] for k, v in results.items()} for i in range(len(results["result"]))]
        reward, terminated = 0, True
        for d, p in zip(dpack, probs):
            r, t = self._apply_action_on_state(d, agent_id, actions[0, agent_id], avail_actions)
            reward += r*p
            terminated = terminated and t
        self.mcts_buffer.update(dpack)
        terminated = terminated or (self.env_t == self.episode_limit)
        
        # Enter episodes to buffer only if test_mode is False
        if not self.terminated:
            transition_data = {
                "obs": obs[0],
                "avail_actions": avail_actions,
                "actions": [(actions[0, agent_id],)],
                "actions_onehot": self._one_hot(self.n_actions, actions[0, agent_id]),
                "reward": [(reward,)],
                "terminated": [(terminated,)]
            }
            
            self.batch.update(transition_data, ts=self.t)
            self.terminated = terminated #or (self.env_t == self.episode_limit)
            self.t += 1

            if terminated and (self.t < self.batch.max_seq_length):
                self.batch.update({
                    "obs": self.get_obs_agent(np.random.choice(self.n_agents))[0][0]
                }, ts=self.t)
            if self.terminated and (not self.test_mode):
                self.buffer.insert_episode_batch(self.batch)

    def get_obs_agent(self, agent_id):
        data = self.mcts_buffer.sample()
        dpack = [{k:v[i] for k, v in data.items()} for i in range(data["probs"].numel())]

        obs = []
        for d in dpack:
            obs.append(self.get_obs_state(d, agent_id))
        return th.stack(obs, dim=0), data

    """ Update env-specific properties of the environment """
    def _update_env_state(self, state):
        return state

    """ Update the previous, stochastic, observation based the new observation
        Generally, the observations, rewards and termination status should be updated (env-specific) """
    def _back_update(self, batch, data, t, ind):
        raise NotImplementedError

    """ Use the general state to create a partial-observability observation for the agents """
    def get_obs_state(self, state, agent_id):
        raise NotImplementedError

    """ Calculate available action in simulated state, default - don't change the env avail_actions """
    def get_avail_actions(self, data, agent_id, avail_actions):
        return avail_actions

    """ Simulate the result of action in the environment """
    def _apply_action_on_state(self, state, agent_id, action, avail_actions, result=0):
        raise NotImplementedError

    """ This function build a dynamic CG using pre-defined (problem specific) model """
    def detect_interaction(self):
        return np.arange(self.n_agents) # by default, no interaction

    def plot_transition(self, t, bs=0):
        raise NotImplementedError

    def _action_results(self, state, agent_id, action):
        raise NotImplementedError

    def _get_data_shape(self, scheme, args):
        return {"state": (scheme["obs"]["vshape"], th.float32)}

    @staticmethod
    def _one_hot(shape, one_hot):
        arr = np.zeros(shape)
        arr[one_hot] = 1
        return arr
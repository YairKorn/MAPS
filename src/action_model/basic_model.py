import numpy as np
import torch as th
from functools import partial
from components.episode_buffer import EpisodeBatch, ReplayBuffer
from components.mcts_buffer import MCTSBuffer
from utils.dict2namedtuple import convert

""" Basic action model class - action models inherit from it """
class BasicAM():
    def __init__(self, scheme, args, stochastic=True) -> None:
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.buffer_size = args.buffer_size
        self.device = "cuda" if args.use_cuda else "cpu"

        # Mechanisms for stochastic environment
        self.stochastic_env = stochastic        # Specified in env-specific action model
        self.action_order = []                  # Order of the agents that performed actions, used to back-updating the observations
        self.apply_MCTS = args.apply_MCTS       # Select approximation method - MCTS (aprox. values) or Mean-state (aprox. state)
        self.MCTS_sampling = args.MCTS_sampling # size of sample every action selection

        mcts_buffer_size = args.MCTS_buffer_size if (self.stochastic_env and self.apply_MCTS) else 1
        self.mcts_buffer = MCTSBuffer(self._get_mcts_scheme(scheme, args), mcts_buffer_size, device=self.device)

        # Episodes management (no need in general, but for compatability with PyMARL)
        self.t_env = 0              # real time of the environment
        self.t = 0                  # time in the episode
        self.test_mode = False      # mode, for preventing training on test episodes
        self.terminated = False     # used to prevent saving steps after the episode was ended
        
        # Additional configuration attributes
        self.decomposed_reward = getattr(args, "decomposed_reward", False)
        self.skip_disabled     = getattr(args, "skip_disabled", False)
        self.default_action    = getattr(args, "default_action", 0)

        # Unpack arguments from sacred
        if getattr(args, "env_args", None):
            args = args.env_args
        if isinstance(args, dict):
            args = convert(args)
        self.args = args

        # Buffer for sequential single-agent samples (rather than n-agents samples)
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

        self.new_batch = partial(EpisodeBatch, model_scheme, groups, 1, self.episode_limit * self.n_agents + 1, preprocess=None)
        self.buffer = ReplayBuffer(model_scheme, groups, self.buffer_size, self.episode_limit * self.n_agents + 1, device=self.device)


    """ When new perception is percepted, update the real state """
    def update_state(self, batch, t_ep, test_mode):
        if t_ep == 0:  # when the env time resets, a new episode has begun
            self.batch = self.new_batch()       # new batch for storing steps
            self.test_mode = test_mode          # does episode is for test or training
            self.t = 0                          # reset internal time
            self.terminated = False

        self.t_env = t_ep                       # update environment real time
        # update state (env-specific method) - used for updating stochastic results and extract state features
        data = self._update_env_state(batch["state"][:, t_ep])    
        self.mcts_buffer.reset(data)

        # In case of stochastic environment, updating the previous observations based on the new observations...
        if self.stochastic_env and (not self.terminated) and (t_ep > 0):
            # ... iterate over agents to update the observation
            for s in range(1, len(self.action_order)):
                self._back_update(self.batch, data, self.t-len(self.action_order)+s, s) # tensors share physical memory
        
        # If decomposed reward is false, re-distribute the reward equally between the agents
        if (not self.decomposed_reward) and t_ep > 0:
            self.batch["reward"][:, (self.t-len(self.action_order)):self.t] = batch["reward"][:, t_ep-1] / len(self.action_order)
        
        self.action_order = [] # reset order of actions
        return data


    """ Update the state based on an action """
    def step(self, agent_id, actions, obs, h_state, avail_actions):
        self.action_order.append(agent_id)

        # MAPS assumpsion: transition function can be decomposed into the product of transition functions of the cliques
        # therefore, sample an example state from MCTS buffer to calculate possible result
        data = self.mcts_buffer.sample(sample_size=1)
        p_result = self._action_results(data, agent_id, actions[0, agent_id]) \
            if self.apply_MCTS else th.tensor([1.])     # If self.apply_MCTS is False, calculate mean-state w.p 1

        results, probs = self.mcts_buffer.mcts_step(p_result, h_state)
        dpack = [{k:v[i] for k, v in results.items()} for i in range(len(results["result"]))]
        reward, terminated = 0, True
        for d, p in zip(dpack, probs):
            r, t = self._apply_action_on_state(d, agent_id, actions[0, agent_id], avail_actions)
            reward += r*p
            terminated = terminated and t
        self.mcts_buffer.update(dpack)
        terminated = terminated or (self.t_env == self.episode_limit)
        
        # Enter episodes to buffer only if test_mode is False
        if not self.terminated:
            transition_data = {
                "obs": obs[0], # arbitrary selects one of the observations to store in buffer
                "avail_actions": avail_actions,
                "actions": [(actions[0, agent_id],)],
                "actions_onehot": self._one_hot(self.n_actions, actions[0, agent_id]),
                "reward": [(reward,)],
                "terminated": [(terminated,)]
            }
            
            self.batch.update(transition_data, ts=self.t)
            self.terminated = terminated
            self.t += 1

            if terminated and (self.t < self.batch.max_seq_length):
                self.batch.update({
                    "obs": self.get_obs_agent(np.random.choice(self.n_agents))[0][0]
                }, ts=self.t)
            if self.terminated and (not self.test_mode):
                self.buffer.insert_episode_batch(self.batch)

    """ For a given agent return observation based on current possible states """
    def get_obs_agent(self, agent_id):
        data = self.mcts_buffer.sample()
        dpack = [{k:v[i] for k, v in data.items()} for i in range(data["probs"].numel())]

        obs = [self.get_obs_state(d, agent_id) for d in dpack]
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
    def detect_interaction(self, data):
        return np.arange(self.n_agents) # by default, no interaction

    """ A debug function (optional) to plot one transition of the environment """
    def plot_transition(self, t, bs=0):
        raise NotImplementedError

    """ A function that get a state and action and return a vector of probabilities of possible outcomes """
    def _action_results(self, state, agent_id, action):
        raise NotImplementedError

    """ State shape for initialization of MCTS buffer """
    def _get_mcts_scheme(self, scheme, args):
        return {
            "state": (scheme["obs"]["vshape"], th.float32),
            "hidden": ((1, args.rnn_hidden_dim), th.float32)
            }

    @staticmethod
    def _one_hot(shape, one_hot):
        arr = np.zeros(shape)
        arr[one_hot] = 1
        return arr
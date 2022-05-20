from enum import unique
import torch as th
import numpy as np
from .TDn_learner import TDnLearner
from components.episode_buffer import EpisodeBatch

class MAPSLearner(TDnLearner):
    def __init__(self, mac, scheme, logger, args):
        super().__init__(mac, scheme, logger, args)
        self.buffer = self.mac.action_model.buffer
        self.args.gamma = np.power(self.args.gamma, 1/self.args.n_agents)

        if self.alg == "TD":
            # TD-n properties
            self.n_bound = args.TDn_bound if args.TDn_bound is not None else args.n_agents+1 # TD-n default n_agents+1
            self.n_values = range(1, self.n_bound+1)
            self.n_weight = th.cat(((1 - args.TDn_weight) * (args.TDn_weight ** th.arange(self.n_bound-1)), \
                th.tensor([args.TDn_weight ** (self.n_bound-1)]))).to(self.device).view(-1, 1, 1, 1)
            print(f'### TD-n Learner uses TD-1...{self.n_bound}, with weights: {self.n_weight[:, 0, 0, 0]} ###')
        elif self.alg == "MC":
            self.n_values[0] = (self.n_values[0] - 1) * args.n_agents + 1
            print(f'### Monte-Carlo Learner uses Conv-{self.n_values} ###')

    """ A learner of MAPS architecture """
    def train(self, _: EpisodeBatch, t_env: int, episode_num: int):
        # use the basic TDn-learner but episodes are taken from the internal, decomposed, buffer
        self.mac.agent.train()
        super().train(self.buffer, t_env, episode_num)
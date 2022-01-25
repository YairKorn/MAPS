from enum import unique
import torch as th
import numpy as np
from .TDn_learner import TDnLearner
from components.episode_buffer import EpisodeBatch

class PSeqLearner(TDnLearner):
    def __init__(self, mac, scheme, logger, args):
        super().__init__(mac, scheme, logger, args)
        self.buffer = self.mac.action_model.buffer
        self.args.gamma = np.power(self.args.gamma, 1/self.args.n_agents)

        # TD-n properties
        self.TDn_bound = args.TDn_bound if args.TDn_bound is not None else args.n_agents+1 # TD-n default n_agents+1
        self.TDn_weight = th.cat(((1 - args.TDn_weight) * (args.TDn_weight ** th.arange(self.TDn_bound-1)), \
            th.tensor([args.TDn_weight ** (self.TDn_bound-1)]))).view(-1, 1, 1, 1)

        print(f'### TDn Learner uses TD-1...{self.TDn_bound}, with weights: {self.TDn_weight[:, 0, 0, 0]}')


    """ A learner of PSeq architecture """
    def train(self, _: EpisodeBatch, t_env: int, episode_num: int):
        # This part is a patch for MCTS: "terminated" may appear more than once in an episode due to inaccurate back_updaing mechanism
        # instead of correcting it, it was patched to allow testing of MCTS
        # ind = th.where(self.buffer["terminated"])
        # unique_ind = th.unique_consecutive(ind[0], return_counts=True)[1]
        # unique_ind = ind[1][th.cat((th.tensor([0]), th.cumsum(unique_ind, dim=0)[:-1]), dim=0)]

        # for b in range(self.buffer.buffer_size):
        #     self.buffer["filled"][b, unique_ind[b]+1:, 0] = 0
        #     self.buffer["terminated"][b, unique_ind[b]+1:, 0] = 0

        # use the basic q-learner but episodes are taken from the internal, decomposed, buffer
        super().train(self.buffer, t_env, episode_num)
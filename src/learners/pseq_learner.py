import torch as th
import numpy as np
from .TDn_learner import TDnLearner
from components.episode_buffer import EpisodeBatch

class PSeqLearner(TDnLearner):
    def __init__(self, mac, scheme, logger, args):
        super().__init__(mac, scheme, logger, args)
        self.buffer = self.mac.action_model.buffer
        self.train_device = "cuda" if args.use_cuda else "cpu" #$ NOT USED FOR NOW
        self.device = self.buffer.device #$ NOT USED FOR NOW
        self.args.gamma = np.power(self.args.gamma, 1/self.args.n_agents)

        # TD-n properties
        self.TDn_bound = args.TDn_bound if args.TDn_bound is not None else args.n_agents+1 # TD-n default n_agents+1
        self.TDn_weight = th.cat(((1 - args.TDn_weight) * (args.TDn_weight ** th.arange(self.TDn_bound-1)), \
            th.tensor([args.TDn_weight ** (self.TDn_bound-1)]))).view(-1, 1, 1, 1)

        print(f'### TDn Learner uses TD-1...{self.TDn_bound}, with weights: {self.TDn_weight[:, 0, 0, 0]}')


    """ A learner of PSeq architecture """
    def train(self, _: EpisodeBatch, t_env: int, episode_num: int):
        # if buffer in CPU and train uses CUDA, move the buffer to CUDA
        if self.buffer.device != self.train_device:
            self.buffer.to(self.train_device)

        # use the basic q-learner but episodes are taken from the internal, decomposed, buffer
        super().train(self.buffer, t_env, episode_num)

        # if buffer in should be in CPU, return the buffer from CUDA to the CPU
        if self.buffer.device != self.device:
            self.buffer.to(self.device)

    # def gamma(self, l=1):
    #     g = th.cat((th.ones((1, self.args.n_agents-1)), self.args.gamma*th.ones((1, 1))), axis=1)
    #     return g.repeat(1, int(l/self.args.n_agents)+1)[0, :l]
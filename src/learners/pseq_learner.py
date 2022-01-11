import torch as th
from .q_learner import QLearner
from components.episode_buffer import EpisodeBatch

class PSeqLearner(QLearner):
    def __init__(self, mac, scheme, logger, args):
        super().__init__(mac, scheme, logger, args)
        self.episode_buffer = self.mac.action_model.buffer
        self.train_device = "cuda" if args.use_cuda else "cpu"
        self.buffer_device = self.episode_buffer.device

    """ A learner of PSeq architecture """
    def train(self, _: EpisodeBatch, t_env: int, episode_num: int):
        # if buffer in CPU and train uses CUDA, move the buffer to CUDA
        if self.episode_buffer.device != self.train_device:
            self.episode_buffer.to(self.train_device)

        # use the basic q-learner but episodes are taken from the internal, decomposed, buffer
        super().train(self.episode_buffer, t_env, episode_num)

        # if buffer in should be in CPU, return the buffer from CUDA to the CPU
        if self.episode_buffer.device != self.buffer_device:
            self.episode_buffer.to(self.buffer_device)

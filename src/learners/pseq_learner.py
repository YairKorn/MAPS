import torch as th
from .q_learner import QLearner
from components.episode_buffer import EpisodeBatch

class PSeqLearner(QLearner):
    def __init__(self, mac, scheme, logger, args):
        super().__init__(mac, scheme, logger, args)
        self.episode_buffer = self.mac.action_model.buffer

    """ A learner of PSeq architecture """
    def train(self, _: EpisodeBatch, t_env: int, episode_num: int):
        # use the basic q-learner but episodes are taken from the internal, decomposed, buffer
        return super().train(self.episode_buffer, t_env, episode_num)
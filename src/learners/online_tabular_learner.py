from enum import unique
import torch as th
import numpy as np
from .tabular_learner import TabularLearner
from components.episode_buffer import EpisodeBatch

class OnlineTabularLearner(TabularLearner):
    def __init__(self, mac, scheme, logger, args):
        pass

    """ A learner of MAPS architecture """
    def train(self, _: EpisodeBatch, t_env: int, episode_num: int):
        pass
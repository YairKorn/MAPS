from collections import defaultdict
from email.policy import default
import torch as th
from collections import defaultdict

class TabularAgent():
    def __init__(self, input_shape, args):
        self.args = args
        self.alpha = args.alpha
        self.n_actions = args.n_actions
        self.qtable = defaultdict(lambda: th.zeros(1, self.n_actions))

    def init_hidden(self):
        pass # irrelevant for tabular

    def forward(self, inputs):
        vs = th.zeros(inputs.shape[0], self.n_actions)

        for i in range(inputs.shape[0]):
            k = inputs[i]
            vs[i, :] = self.qtable[k]
        return vs
    
    def update_qvalues(self, obs, actions, targets):
        for i in range(obs.shape[0]):
            for j in range(obs.shape[1]):
                for k in range(obs.shape[2]):
                    obs_k = obs[i, j, k]
                    if obs_k not in self.qtable:
                        self.qtable[obs_k] = th.zeros(1, self.n_actions)
                    self.qtable[obs_k][0, actions[i, j, k, 0]] = targets[i, j, k]

        pass
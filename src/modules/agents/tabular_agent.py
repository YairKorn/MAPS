from collections import defaultdict
from email.policy import default
import torch as th
from collections import defaultdict

class TabularAgent():
    def __init__(self, input_shape, args):
        self.args = args
        self.n_actions = args.n_actions
        self.qtable = defaultdict(lambda: th.zeros(1, self.n_actions))

    def init_hidden(self):
        pass # irrelevant for tabular

    def forward(self, inputs, hidden_state):
        vs = th.zeros(inputs.shape[0], self.n_actions)

        for i in inputs.shape[0]:
            k = hash(inputs(i))
            vs[i, :] = self.qtable[k]
        return vs
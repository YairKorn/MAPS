import pickle
import torch as th
import os
os.environ["PYTHONHASHSEED"] = "0"

class TabularAgent():
    def __init__(self, input_shape, args):
        self.args = args
        self.alpha = args.alpha
        self.n_actions = args.n_actions
        self.qtable = {}

    def init_hidden(self):
        pass # irrelevant for tabular

    def forward(self, inputs):
        vs = th.zeros(inputs.shape[0], inputs.shape[1], self.n_actions)

        for i in range(inputs.shape[0]):
            for j in range(inputs.shape[1]):
                k = hash(tuple(inputs[i, j].tolist()))
                if k in self.qtable:
                    vs[i, j, :] = self.qtable[k]
        return vs
    
    def update_qvalues(self, obs, actions, targets):
        for i in range(obs.shape[0]):
            for j in range(obs.shape[1]):
                for k in range(obs.shape[2]):
                    if targets[i, j, k] > -1e6:
                        obs_k = hash(tuple(obs[i, j, k].tolist()))
                        if obs_k not in self.qtable:
                            self.qtable[obs_k] = th.zeros(1, self.n_actions)
                        self.qtable[obs_k][0, actions[i, j, k, 0]] = targets[i, j, k]


    def save_model(self, path):
        with open(os.path.join(path, "qtable.txt"), 'wb+') as f:
            pickle.dump(self.qtable, f)

    def load_model(self, path):
        with open(os.path.join(path, "qtable.txt"), 'rb') as f:
            self.qtable = pickle.load(f)
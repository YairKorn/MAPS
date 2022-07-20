import pickle
import torch as th
import numpy as np
import os
os.environ["PYTHONHASHSEED"] = "0"

class OnlineTabularAgent():
    def __init__(self, input_shape, args):
        self.args = args
        self.n_actions = args.n_actions
        self.qtable = {}

        self.alpha = args.alpha
        self.gamma = np.power(self.args.gamma, 1/self.args.n_agents)

    def init_hidden(self):
        pass # irrelevant for tabular

    def forward(self, inputs):
        vs = th.zeros(inputs.shape[0], self.n_actions)

        for i in range(inputs.shape[0]):
            k = hash(tuple(inputs[i].tolist()))
            if k in self.qtable:
                vs[i, :] = self.qtable[k]
        return vs
    
    
    def update_qvalues(self, states, actions, avail_actions, rewards):
        for i in range(states.shape[0]):
            lstate = [hash(tuple(s.tolist())) for s in states[i]]

            for j in range(len(lstate)-2, -1, -1):
                if lstate[j+1] in self.qtable:
                    values = self.qtable[lstate[j+1]].clone()
                    values[0, avail_actions[i, j+1] == 0] = -1e7
                else:
                    values = th.zeros(1, self.n_actions)

                
                if lstate[j] not in self.qtable:
                    self.qtable[lstate[j]] = th.zeros(1, self.n_actions)
                qvalue = self.qtable[lstate[j]]

                target = (1 - self.alpha) * qvalue[0, actions[i, j]] + self.alpha * \
                    (rewards[i, j] + self.gamma * th.max(values))
                self.qtable[lstate[j]][0, actions[i, j]] = target


    def save_model(self, path):
        with open(os.path.join(path, "qtable.txt"), 'wb+') as f:
            pickle.dump(self.qtable, f)

    def load_model(self, path):
        with open(os.path.join(path, "qtable.txt"), 'rb') as f:
            self.qtable = pickle.load(f)
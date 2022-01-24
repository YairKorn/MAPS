from numpy.random.mtrand import sample
import torch as th
import numpy as np
from types import SimpleNamespace as SN

class MCTSBuffer:
    def __init__(self, scheme, max_size, dtype=th.float32, device="cpu") -> None:
        # Attributes
        self.max_size = max_size        
        self.device = device

        # Create data structure
        self.scheme = scheme
        self.data = SN()
        self.data = {"probs": th.cat((th.tensor([1], device=self.device), th.zeros(max_size-1, device=self.device)))}

        assert "state" in scheme, "state shape must be in scheme"
        for k, v in scheme.items():
            self.data[k] = th.zeros((max_size,) + v[0], dtype=v[1], device=self.device)
        self.filled = 1 # count number of filled slots, start with an "empty state"


    def sample(self, sample_size=0):
        # sample size 0 means to get all available states
        sample = np.arange(self.filled) if (not sample_size) else \
             np.random.choice(self.filled, size=sample_size, replace=False, p=self.data["probs"][:self.filled].numpy())

        return {k:self.data[k][sample] for k in self.data.keys()}


    # Sample from the possible results of an action, based on the action model, and return batch of states
    def mcts_step(self, v_results):
        # Calculate probabilities of all possible outcomes; mask 
        results = (self.data["probs"][:self.filled].view(-1, 1) @ v_results.view(1, -1)).reshape(-1)

        # If the batch is smaller than the maximal batch size, sample randomly (MC)
        sample = th.where(results > 0)[0] if (results > 0).sum() <= self.max_size else \
            th.tensor(np.random.choice(results.numel(), size=min(self.max_size, self.filled), replace=False, p=results.numpy()))

        self.filled = sample.numel()
        self.data["probs"][:self.filled] = results[sample] / results[sample].sum()

        ret_data = {k:self.data[k][(sample/v_results.numel()).long()] for k in self.scheme.keys()}
        ret_data["result"] = [x%v_results.numel() for x in sample]
        return ret_data, self.data["probs"]


    # Update states based corresponds to the probabilities calculated in the sampling
    def update(self, data):
        assert len(data) <= self.max_size, "Too many values to update"

        for k in self.scheme.keys():
            self.data[k][:len(data), :] = th.stack([d[k] for d in data], dim=0)
        self.filled = len(data)
    

    # Reset buffer, including save only one (known) state and delete all other states
    def reset(self, data):
        for k in self.scheme.keys():
            self.data[k][0] = data[k].reshape(self.scheme[k][0])
        self.data["probs"][0] = 1.0
        self.filled = 1
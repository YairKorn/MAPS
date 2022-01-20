import torch as th
import numpy as np
from types import SimpleNamespace as SN

class MCTSBuffer:
    def __init__(self, state_size, max_size, dtype=th.float32, device="cpu") -> None:
        # Attributes
        self.max_size = max_size        
        self.device = device

        # Create data structure ###! NEED TO RESET BUFFER - CAN BE DONE USING FILLED FIELD
        self.data = SN()
        self.data = {
            "state": th.zeros((max_size, state_size), dtype=dtype, device=self.device),
            "probs": th.zeros((max_size, 1), dtype=th.float32, device=self.device),
            "filled": th.zeros((max_size, 1), dtype=th.long, device=self.device)
        }
    
    # Sample from the possible results of an action, based on the action model, and return batch of states
    def action_sample(self, v_results):
        # Calculate probabilities of all possible outcomes; mask 
        results = th.tensordot(self.data["probs"]*self.data["filled"], v_results, dims=0).reshape(-1)

        # If the batch is smaller than the maximal batch size, sample randomly (MC)
        sample = th.where(results > 0) if (results > 0).sum() > self.max_size else \
            np.random.choice(results.size, size=self.max_size, replace=False, p=results) #! to numpy?

        self.data["probs"][0, :sample.size] = results[sample]
        return [(x/v_results.size, x%v_results.size) for x in sample]
    
    # Update states based corresponds to the probabilities calculated in the sampling
    def update(self, data):
        self.data["state"][:len(data), :] = th.stack(data, dim=1)
        self.data["filled"][0, :len(data)] = 1
    
    # Reset buffer by set all spots to be empty
    def reset(self):
        self.data["filled"] = 0
#import torch as th
from torch.optim import RMSprop, Adam

class Optimizer():
    def __init__(self, opt_type, params, lr, **kwargs) -> None:
        if opt_type == "RMSProp":
            self.optimizer = RMSprop(params, lr, alpha=kwargs['alpha'], eps=kwargs['eps'])
        elif opt_type == "Adam":
            self.optimizer = Adam(params, lr, eps=kwargs['eps'])

    def __get__(self):
        return self.optimizer

import numpy as np

""" Basic action model class - action models inherit from it """
class ActionModel():
    def __init__(self, **kwargs) -> None:
        args = kwargs

        # stochastic environment properties
        self.stochastic_env  = getattr(args, "stochastic_environment", False)
        self.monte_carlo     = getattr(args, "apply_monte_carlo", False)
        self.mcts_batch_size = getattr(args, "mcts_batch_size", 32)

    """ When new perception is percepted, update the real state """
    def update_state(self, state):
        self.states = StateUnit(state, 1.0)

    """ Update the state based on an action """
    def step(self, action):
        raise NotImplementedError

    # """ Calculate new states based on their probability """
    # def mcts(self):
    #     pass


class StateUnit():
    def __init__(self, state, prob) -> None:
        self.state = state
        self.prob  = prob
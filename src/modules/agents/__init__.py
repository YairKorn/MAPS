REGISTRY = {}

from .rnn_agent import RNNAgent
REGISTRY["rnn"] = RNNAgent

from .rnn_feature_agent import RNNFeatureAgent
REGISTRY["rnn_feat"] = RNNFeatureAgent

from .ddqn_agent import DDQNAgent
REGISTRY["dual"] = DDQNAgent
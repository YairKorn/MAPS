from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner

from .dcg_learner import DCGLearner
REGISTRY["dcg_learner"] = DCGLearner

from .maps_learner import MAPSLearner
REGISTRY["maps_learner"] = MAPSLearner

from .maps_tabular_learner import MAPSTabularLearner
REGISTRY["tabular_learner"] = MAPSTabularLearner
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

from .online_tabular_learner import OnlineTabularLearner
REGISTRY["ol_tabular_learner"] = OnlineTabularLearner

from .q_tabular_learner import QTabularLearner
REGISTRY["tabular_learner"] = QTabularLearner
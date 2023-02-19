from functools import partial
from smac.env import MultiAgentEnv, StarCraft2Env
import sys
import os

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))

from .stag_hunt import StagHunt
REGISTRY["stag_hunt"] = partial(env_fn, env=StagHunt)

from .adv_coverage import AdversarialCoverage
REGISTRY["mrac"] = partial(env_fn, env=AdversarialCoverage)

from .het_adv_coverage import HeterogeneousAdversarialCoverage
REGISTRY["het_mrac"] = partial(env_fn, env=HeterogeneousAdversarialCoverage)

from .hunt_trip import HuntingTrip
REGISTRY["hunt_trip"] = partial(env_fn, env=HuntingTrip)
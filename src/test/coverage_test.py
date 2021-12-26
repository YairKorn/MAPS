import yaml, pytest, os
from src.utils.dict2namedtuple import convert
from src.envs.coverage import AdversarialCoverage

CONFIG_PATH = os.path.join('.', 'src', 'test', 'coverage_test.yaml')
with open(CONFIG_PATH, 'r') as config:
    args = yaml.safe_load(config)['env_args']

class TestCoverage:

    def test_init(self):
        env = AdversarialCoverage(**args)
        assert env.grid.size == env.width * env.height * 3
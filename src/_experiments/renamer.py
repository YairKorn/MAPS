import os, json
from datetime import datetime
from argparse import ArgumentParser
PATH = os.path.join(os.getcwd(), 'results', 'sacred')

def rename_dir(skip=2):
    experiments = [x[0] for x in os.walk(PATH)]
    experiments = list(filter(lambda e: e.split('/')[-1].isdigit(), experiments))  # Remove already-renamed exp
    experiments.sort()

    for e in experiments[:len(experiments)-skip]:
        with open(os.path.join(e, 'config.json')) as f:
            config = json.load(f)

        # New name: _env#_alg#_date
        _env  = config["env_args"]["map"]
        _alg  = config["name"].upper()
        _date = datetime.fromtimestamp(os.path.getctime(e)).strftime("%Y-%m-%d-%H:%M:%S")
    
        _name = '#'.join([_env, _alg, _date])
        os.rename(e, os.path.join(PATH, _name))


def make_parser():
    parser = ArgumentParser(description="Arguments for rename experiments")
    parser.add_argument('skip', type=int, nargs='?', default=2)
    return parser

if __name__ == '__main__':
    args = make_parser().parse_args()
    rename_dir(args.skip)

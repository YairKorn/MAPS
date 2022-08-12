import os, json, shutil
from datetime import datetime
from argparse import ArgumentParser
PATH = os.path.join(os.getcwd(), 'results', 'sacred')

def rename_dir(skip=2, prefix=''):
    if prefix: # Update prefix
        prefix = prefix + '_'

    dirs = [name for name in os.listdir(PATH) if os.path.isdir(os.path.join(PATH, name))]
    dirs = list(filter(lambda e: e.isdigit(), dirs))
    dirs.sort(key=int)

    experiments = [os.path.join(PATH, d) for d in dirs[:len(dirs)-skip]]

    for e in experiments:
        with open(os.path.join(e, 'config.json')) as f:
            config = json.load(f)

        # New name: _env#_alg#_date
        _path = os.path.join("results", "sacred", config["env_args"]["map"])
        if not os.path.exists(_path):
            os.mkdir(_path)

        _alg  = prefix + config["name"].upper()
        _date = datetime.fromtimestamp(os.path.getctime(e)).strftime("%Y-%m-%d-%H:%M:%S")
        _name = '#'.join([_alg, _date])
        os.rename(e, os.path.join(PATH, _name))
    
        shutil.move(os.path.join(PATH, _name), os.path.join(_path, _name))
        # os.rename(e_new, os.path.join(_path, _name))

def make_parser():
    parser = ArgumentParser(description="Arguments for rename experiments")
    parser.add_argument('skip', type=int, nargs='?', default=2)
    parser.add_argument('prefix', type=str, default='')
    return parser

if __name__ == '__main__':
    args = make_parser().parse_args()
    rename_dir(args.skip, args.prefix)
    # rename_dir(0, '') #! DEBUG
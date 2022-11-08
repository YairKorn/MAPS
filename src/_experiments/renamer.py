import os, json, shutil, collections
from datetime import datetime
from argparse import ArgumentParser
PATH = os.path.join(os.getcwd(), 'results', 'sacred')

# Function for flatten a dictionary
def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def rename_dir(skip=2, prefix='', fields=None):
    if prefix: # Update prefix
        prefix = prefix + '_'

    dirs = [name for name in os.listdir(PATH) if os.path.isdir(os.path.join(PATH, name))]
    dirs = list(filter(lambda e: e.isdigit(), dirs))
    dirs.sort(key=int)

    experiments = [os.path.join(PATH, d) for d in dirs[:len(dirs)-skip]]

    for e in experiments:
        try:
            with open(os.path.join(e, 'config.json')) as f:
                config = json.load(f)

            # New name: _alg#_args#_date
            _path = os.path.join("results", "sacred", config["env_args"]["map"])
            if not os.path.exists(_path):
                os.mkdir(_path)

            _alg  = prefix + config["name"].upper()
            _date = datetime.fromtimestamp(os.path.getctime(e)).strftime("%Y-%m-%d-%H:%M:%S")
            
            config = flatten(config)
            if fields:
                _args = []
                for f in fields:
                    tmp_arg, tmp_name = f.split('=') if '=' in f else (f, f)
                    _args.append(f'{tmp_name}={config[tmp_arg]}' if tmp_arg in config else '')
                    # _args = '&'.join([f'{f}={config[f]}' if f in config else '' for f in fields])
                _args = '&'.join(_args)
                _name = '#'.join([_alg, _args, _date])
            else:
                _name = '#'.join([_alg, _date])
            
            os.rename(e, os.path.join(PATH, _name))
        
            shutil.move(os.path.join(PATH, _name), os.path.join(_path, _name))
            # os.rename(e_new, os.path.join(_path, _name))
        except:
            print(f"Failed to rename {e}")

def make_parser():
    parser = ArgumentParser(description="Arguments for rename experiments")
    parser.add_argument('skip', type=int, nargs='?', default=2)
    parser.add_argument('prefix', type=str, default='')
    parser.add_argument('--f', '--list', nargs='+') # Categories
    return parser

if __name__ == '__main__':
    args = make_parser().parse_args()
    rename_dir(args.skip, args.prefix, args.f)
    # rename_dir(0, '', ['env_args_reduced_decay=decay']) #! DEBUG
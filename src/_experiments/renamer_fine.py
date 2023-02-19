import os, json, shutil
from _collections_abc import MutableMapping
from datetime import datetime
from argparse import ArgumentParser
PATH = os.path.join(os.getcwd(), 'results', 'sacred')


def rename_dir(folder='', keys=None):
    if folder == '':
        raise "Invalid path"

    if keys:
        keys = {k.split('@')[0] : k.split('@')[1] for k in keys}
    path = os.path.join(PATH, folder)

    folders = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

    for f in folders:
        if f[:f.rfind('#')] in keys:
            new_name = keys[f[:f.rfind('#')]] + '#' + f[f.rfind('#')+1:]
            
            shutil.move(os.path.join(path, f), os.path.join(path, new_name))


def make_parser():
    parser = ArgumentParser(description="Arguments for rename experiments")
    parser.add_argument('folder', type=str, nargs='?', default=2)
    parser.add_argument('--keys', '--list', nargs='+')
    return parser

if __name__ == '__main__':
    args = make_parser().parse_args()
    rename_dir(args.folder, args.keys)
    # rename_dir('Z7_Peripherial1', ['PPO#alpha=-0.2&reward=0.2&decay=0.6&shape=decay:PPO']) #! DEBUG



"""
Examples:
python src/_experiments/renamer_fine.py 'Z8_Necessary1_1M_alpha=0.05' --keys 'DQL_TABULAR#alpha=-0.05&reward=0.2&decay=0.6&shape=decay@Tabular_QL' 'IQL#alpha=-0.05&reward=0.2&decay=0.6&shape=decay@IQL' 'PPO#alpha=-0.05&reward=0.2&decay=0.6&shape=decay@PPO'
python src/_experiments/renamer_fine.py 'Z8_Necessary1_1M_alpha=0.2' --keys 'DQL_TABULAR#alpha=-0.2&reward=0.2&decay=0.6&shape=decay@Tabular_QL' 'IQL#alpha=-0.2&reward=0.2&decay=0.6&shape=decay@IQL' 'PPO#alpha=-0.2&reward=0.2&decay=0.6&shape=decay@PPO'


"""
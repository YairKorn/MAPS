from cProfile import label
import os, json
from turtle import color
import types
from attr import field
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser

fields = ['loss', 'target_mean', 'td_error_abs', 'q_taken_mean', 'grad_norm']
colors = ['g', 'b', 'c', 'm', 'k']

def plot_run(name, mode):
    # Read data from file
    path = os.path.join(os.getcwd(), 'results', 'sacred', name, 'info.json')
    data = json.load(open(path, 'r'))

    time = np.asarray(data["test_return_mean_T"])

    fig, axs = plt.subplots(2)

    test_mean = np.asarray([d["value"] for d in data["test_return_mean"]])
    test_std  = np.asarray([d["value"] for d in data["test_return_std"]])
    axs[0].plot(time, test_mean, color='red', label='reward')
    axs[0].fill_between(time, test_mean - test_std, test_mean + test_std, color='red', alpha=.25)

    if mode:
        time = np.asarray(data['loss_T'])
        for f, c in zip(fields, colors):
            dfield = np.asarray(data[f])
            axs[1].plot(time, dfield, color=c, label=f)
        plt.legend(loc="best")

    fig.tight_layout()
    plt.suptitle(name)
    plt.show()


def make_parser():
    parser = ArgumentParser(description="Arguments for analizing an experiment")
    parser.add_argument('name', type=str)
    parser.add_argument('mode', type=bool, nargs='?', default=True) # True for deep, False for tabular
    return parser

# For running: python src/_experiments/analize_run.py DIR
if __name__ == '__main__':
    args = make_parser().parse_args()
    plot_run(args.name, args.mode)
    # plot_run('Threat_7x7_Hard##MAPS*#2022-02-15-23:09:31')
import os, sys, json
from turtle import color
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
COLORS = ['blue', 'red', 'black', 'green', 'purple', 'tan', 'lime', 'skyblue', 'fuchsia', 'grey']

def visualize(prefix, window=1, c=None, metric=None):
    # list of file to plot on graph
    experiments = os.listdir(os.path.join(os.getcwd(), 'results', 'sacred', prefix))
    experiments.sort()

    if c is None:
        c = experiments

    # Compress dataset into categories
    dataset = {}
    for e in experiments:
        prefixes = [e.startswith(cat)*len(cat) for cat in c]
        match_pre = prefixes.index(max(prefixes))

        if max(prefixes): # Ignore data points that not realated to any category
            if c[match_pre] in dataset:
                dataset[c[match_pre]].append(e)
            else:
                dataset[c[match_pre]] = [e]

    # Create figure, unifie dataset and plot it
    fig, axs = plt.subplots(len(metric))
    fig.suptitle(prefix, fontweight ='bold', fontsize=36, fontname="Ubuntu")
    for i in range(len(metric)):
        axs[i].set_title(metric[i], fontsize=30)


    for pre, exp in dataset.items():
        # Extract the first file data, which used as reference to other files
        data = json.load(open(os.path.join(os.getcwd(), 'results', 'sacred', prefix, exp[0], "info.json"), 'r'))
        dtime = [np.asarray(data[m + "_T"]) for m in metric] #! NOTE: doesn't work if there are different scales of measurements
        if window > 1:
            for i in range(len(dtime)):
                dtime[i] = dtime[i][:-window+1]

        agr_data = [np.convolve(np.asarray(data[m]), np.ones(window), 'valid') / window for m in metric]
        cnt_data = [1] * len(metric)

        # Calculate average result of this category
        if len(exp) > 1:
            for dir in exp[1:]:
                data = json.load(open(os.path.join(os.getcwd(), 'results', 'sacred', prefix, dir, "info.json"), 'r'))

                for i in range(len(metric)):
                    if metric[i] in data:
                        # Add data
                        tmp_data = np.convolve(np.asarray(data[metric[i]]), np.ones(window), 'valid')
                        if agr_data[i].size > tmp_data.size:
                            tmp_data = np.pad(tmp_data, (0, agr_data[i].size - tmp_data.size), 'constant', constant_values=tmp_data[-1])
                        agr_data[i] += tmp_data[:agr_data[i].size]

                        # Aggregate the number of times
                        cnt_data[i] += 1


        # average over all results
        for i in range(len(metric)):
            agr_data[i] /= cnt_data[i]

        for i in range(len(metric)):
            axs[i].plot(dtime[i], agr_data[i], COLORS[0], label=pre)
        COLORS.append(COLORS.pop(0))

    # plt.title(prefix)
    plt.legend(loc="best", fontsize=30)
    # plt.title(prefix,)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()



def make_parser():
    parser = ArgumentParser(description="Arguments for graph creation")
    parser.add_argument('prefix', type=str)
    parser.add_argument('window', type=int, nargs='?', default=1)
    parser.add_argument('--c', '--list', nargs='+') # Categories
    # parser.add_argument('--m', '--list', nargs='+') # Metrics
    return parser

# Run using command line:
# python plot.py prefix (window) --c cat1 cat2 etc.
# 
if __name__ == "__main__":
    # Experiments on the same environment and plotted together
    args = make_parser().parse_args()
    visualize(args.prefix, args.window, args.c, ['variance', 'entropy'])
    # visualize('T7x7_Medium_4A', 200, ['ERND_MAPS', 'EDIR_MAPS'], ['variance', 'entropy']) # Debug

    #! Complete: Flexible choose of variables
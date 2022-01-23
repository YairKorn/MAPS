import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
COLORS = ['blue', 'red', 'black', 'green', 'purple', 'tan', 'lime', 'skyblue', 'fuchsia', 'grey']

def visualize(prefix, window=1, c=None):
    # list of file to plot on graph
    experiments = [x[0] for x in os.walk(os.path.join(os.getcwd(), 'results', 'sacred'))]    
    experiments = list(filter(lambda x: x.split('/')[-1].startswith(prefix), experiments))
    labels = [x.split('/')[-1].replace(prefix + '_', '') for x in experiments]

    if c is None:
        c = labels

    # Compress dataset into categories
    dataset = {}
    for l, e in zip(labels, experiments):
        prefixes = [l.startswith(cat)*len(cat) for cat in c]
        match_pre = prefixes.index(max(prefixes))

        if max(prefixes): # Ignore data points that not realated to any category
            if c[match_pre] in dataset:
                dataset[c[match_pre]].append(e)
            else:
                dataset[c[match_pre]] = [e]

    # Create figure, unifie dataset and plot it
    plt.figure()

    for pre, exp in dataset.items():
        # Extract the first file data, which used as reference to other files
        data = json.load(open(os.path.join(exp[0], "info.json"), 'r'))
        test_time = np.asarray(data["test_return_mean_T"])
        if window > 1:
            test_time = test_time[:-window+1]

        test_mean = np.expand_dims(np.convolve(np.asarray([d["value"] for d in data["test_return_mean"]]), \
            np.ones(window), 'valid') / window, axis=0)
        
        test_std = np.expand_dims(np.convolve(np.asarray([d["value"] for d in data["test_return_std"]]), \
            np.ones(window), 'valid') / window, axis=0)

        # Calculate average result of this category
        if len(exp) > 1:
            for dir in exp[1:]:
                data = json.load(open(os.path.join(dir, "info.json"), 'r'))

                m_data = np.convolve(np.asarray([d["value"] for d in data["test_return_mean"]]), np.ones(window), 'valid') / window
                m_data = m_data[:min(test_time.size, m_data.size)] # Trim longer data
                m_data = np.pad(m_data, (0, test_time.size - m_data.size), 'constant', constant_values=m_data[-1])

                test_mean = np.concatenate((test_mean, np.expand_dims(m_data, axis=0)), axis=0)

                s_data = np.convolve(np.asarray([d["value"] for d in data["test_return_std"]]), np.ones(window), 'valid') / window
                s_data = s_data[:min(test_time.size, s_data.size)] # Trim longer data
                s_data = np.pad(s_data, (0, test_time.size - s_data.size), 'constant', constant_values=s_data[-1]) # Pad shorter data

                test_std = np.concatenate((test_std, np.expand_dims(s_data, axis=0)), axis=0)

        # average over all results
        test_mean = np.average(test_mean, axis=0)
        test_std = np.average(test_std, axis=0)

        plt.plot(test_time, test_mean, COLORS[0], label=pre)
        plt.fill_between(test_time, test_mean - test_std, test_mean + test_std, alpha=.25)
        COLORS.append(COLORS.pop(0))

    plt.title(prefix)
    plt.legend(loc="best")
    plt.show()


def make_parser():
    parser = ArgumentParser(description="Arguments for graph creation")
    parser.add_argument('prefix', type=str)
    parser.add_argument('window', type=int, nargs='?', default=1)
    parser.add_argument('--c', '--list', nargs='+')
    return parser

# Run using command line:
# python plot.py prefix (window) --c cat1 cat2 etc.
# 
if __name__ == "__main__":
    # Experiments on the same environment and plotted together
    args = make_parser().parse_args()
    visualize(args.prefix, args.window, args.c)
    # visualize('Cov_11x11_9Boxes', 1, ['PSeq_Obs']) # Debug
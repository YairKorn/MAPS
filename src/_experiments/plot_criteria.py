import os, sys, json
from turtle import color
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
COLORS = ['blue', 'red', 'black', 'green', 'purple', 'tan', 'lime', 'skyblue', 'fuchsia', 'grey']


def visualize(prefix, window=1, c=None):
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
    plt.figure()

    for pre, exp in dataset.items():
        # Extract the first file data, which used as reference to other files
        data = json.load(open(os.path.join(os.getcwd(), 'results', 'sacred', prefix, exp[0], "info.json"), 'r'))
        if 'criteria_mean' in data:
            test_time = np.asarray(data["criteria_mean_T"])
            if window > 1:
                test_time = test_time[:-window+1]

            criteria_mean = np.expand_dims(np.convolve(np.asarray([d["value"] for d in data["criteria_mean"]]), \
                np.ones(window), 'valid') / window, axis=0)

            criteria_std = np.expand_dims(np.convolve(np.asarray([d["value"] for d in data["criteria_std"]]), \
                np.ones(window), 'valid') / window, axis=0)

            # Calculate average result of this category
            if len(exp) > 1:
                for dir in exp[1:]:
                    try:
                        # data = json.load(open(os.path.join(dir, "info.json"), 'r'))
                        data = json.load(open(os.path.join(os.getcwd(), 'results', 'sacred', prefix, dir, "info.json"), 'r'))

                        m_data = np.convolve(np.asarray([d["value"] for d in data["criteria_mean"]]), np.ones(window), 'valid') / window
                        m_data = m_data[:min(test_time.size, m_data.size)] # Trim longer data
                        m_data = np.pad(m_data, (0, test_time.size - m_data.size), 'constant', constant_values=m_data[-1])

                        test_mean = np.concatenate((test_mean, np.expand_dims(m_data, axis=0)), axis=0)

                        s_data = np.convolve(np.asarray([d["value"] for d in data["criteria_std"]]), np.ones(window), 'valid') / window
                        s_data = s_data[:min(test_time.size, s_data.size)] # Trim longer data
                        s_data = np.pad(s_data, (0, test_time.size - s_data.size), 'constant', constant_values=s_data[-1]) # Pad shorter data

                        test_std = np.concatenate((test_std, np.expand_dims(s_data, axis=0)), axis=0)
                    except:
                        pass

            # average over all results
            criteria_mean = np.average(criteria_mean, axis=0)
            criteria_std = np.average(criteria_std, axis=0)

            plt.plot(test_time, criteria_mean, COLORS[0], label=pre)
            plt.fill_between(test_time, criteria_mean - criteria_std, criteria_mean + criteria_std, color=COLORS[0], alpha=.25)
            COLORS.append(COLORS.pop(0))

    plt.title(prefix + " Criteria", fontweight ='bold', fontsize=36, fontname="Ubuntu")
    plt.legend(loc="best", fontsize=36)
    plt.xlabel("Time Steps", fontsize=28)
    plt.ylabel("Optimization Value", fontsize=28)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.show()


def make_parser():
    parser = ArgumentParser(description="Arguments for graph creation")
    parser.add_argument('prefix', type=str)
    parser.add_argument('window', type=int, nargs='?', default=1)
    parser.add_argument('--c', '--list', nargs='+')
    return parser


# Run using command line:
# python plot.py prefix (window) --c cat1 cat2 etc.
if __name__ == "__main__":
    # Experiments on the same environment and plotted together
    args = make_parser().parse_args()
    visualize(args.prefix, args.window, args.c)
    # visualize('S7x7_4Boxes', 1) # Debug

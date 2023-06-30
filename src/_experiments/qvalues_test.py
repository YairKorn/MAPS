import os, sys, json
from turtle import color
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
COLORS = ['blue', 'red', 'black', 'green', 'purple', 'tan', 'lime', 'skyblue', 'fuchsia', 'grey']

def visualize(window=1, c=None):
    experiments = {f'QTest_{i}Agents': os.listdir(os.path.join(os.getcwd(), 'results', 'sacred', f'QTest_{i}Agents')) for i in range(3,11)}

    if c is None:
        c = experiments

    # Create figure, unifie dataset and plot it
    plt.figure()

    # Compress dataset into categories
    for env in experiments.keys():
        dataset = {}
        for e in experiments[env]:
            prefixes = [e.startswith(cat)*len(cat) for cat in c]
            match_pre = prefixes.index(max(prefixes))

            if max(prefixes): # Ignore data points that not realated to any category
                if c[match_pre] in dataset:
                    dataset[c[match_pre]].append(e)
                else:
                    dataset[c[match_pre]] = [e]

        for pre, exp in dataset.items():
            # Extract the first file data, which used as reference to other files
            data = json.load(open(os.path.join(os.getcwd(), 'results', 'sacred', env, exp[0], "info.json"), 'r'))
            q_time = np.asarray(data["q_loss_T"])
            if window > 1:
                q_time = q_time[:-window+1]

            q_data = np.convolve(np.asarray(data['q_loss']), np.ones(window), 'valid') / window
            cnt_data = 1

            # Calculate average result of this category
            if len(exp) > 1:
                for dir in exp[1:]:
                    try:
                        # data = json.load(open(os.path.join(dir, "info.json"), 'r'))
                        data = json.load(open(os.path.join(os.getcwd(), 'results', 'sacred', env, dir, "info.json"), 'r'))

                        tmp_data = np.convolve(np.asarray(data['q_loss']), np.ones(window), 'valid') / window
                        if q_data.size > tmp_data.size:
                            tmp_data = np.pad(tmp_data, (0, q_data.size - tmp_data.size), 'constant', constant_values=tmp_data[-1])
                        cnt_data += 1

                    except:
                        pass

            # average over all results
            q_data /= cnt_data

            lstyle = '--' if pre=='VDN' else '-.'
            plt.plot(q_time, q_data, color=COLORS[0], linestyle=lstyle, label=env + '_' + pre)
        
        COLORS.append(COLORS.pop(0))

    plt.title('QTest_Results', fontweight ='bold', fontsize=36, fontname="Ubuntu")
    plt.legend(loc="best", fontsize=18)
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
# 
if __name__ == "__main__":
    # Experiments on the same environment and plotted together
    # args = make_parser().parse_args()
    # visualize(args.prefix, args.window, args.c)
    visualize(1, ['MAPS', 'VDN']) # Debug
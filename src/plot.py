import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
COLORS = ['b', 'r', 'g', 'c', 'k', 'm']

def visualize(prefix):
    # list of file to plot on graph
    experiments = [x[0] for x in os.walk(os.path.join(os.getcwd(), 'results', 'sacred'))]    
    experiments = list(filter(lambda x: x.split('/')[-1].startswith(prefix), experiments))
    labels = [x.split('/')[-1].replace(prefix + '_', '') for x in experiments]

    plt.figure()
    for dir, label in zip(experiments, labels):
        data = json.load(open(os.path.join(dir, "info.json"), 'r'))
        
        test_time = np.asarray(data["test_return_mean_T"])
        test_mean = np.asarray([d["value"] for d in data["test_return_mean"]])
        test_std  = np.asarray([d["value"] for d in data["test_return_std"]])
  
        plt.plot(test_time, test_mean, COLORS[0], label=label)
        plt.fill_between(test_time, test_mean - test_std, test_mean + test_std, alpha=.25)
    
        # Switch color for the next graph
        COLORS.append(COLORS.pop(0))

    plt.legend(loc="best")
    plt.show()

if __name__ == "__main__":
    # Experiments on the same environment and plotted together
    prefix = sys.argv[1] if len(sys.argv) > 1 else ''
    visualize(prefix)
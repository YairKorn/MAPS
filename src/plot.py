import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
COLORS = ['blue', 'red', 'black', 'green', 'purple', 'tan', 'lime', 'skyblue', 'fuchsia', 'grey']

def visualize(prefix, window=1):
    # list of file to plot on graph
    experiments = [x[0] for x in os.walk(os.path.join(os.getcwd(), 'results', 'sacred'))]    
    experiments = list(filter(lambda x: x.split('/')[-1].startswith(prefix), experiments))
    labels = [x.split('/')[-1].replace(prefix + '_', '') for x in experiments]

    plt.figure()
    for dir, label in zip(experiments, labels):
        data = json.load(open(os.path.join(dir, "info.json"), 'r'))
        
        test_time = np.asarray(data["test_return_mean_T"])[:-window+1]
        test_mean = np.asarray([d["value"] for d in data["test_return_mean"]])
        test_mean = np.convolve(test_mean, np.ones(window), 'valid') / window
        
        test_std  = np.asarray([d["value"] for d in data["test_return_std"]])
        test_std  = np.convolve(test_std, np.ones(window), 'valid') / window
  
        plt.plot(test_time, test_mean, COLORS[0], label=label)
        plt.fill_between(test_time, test_mean - test_std, test_mean + test_std, alpha=.25)
    
        # Switch color for the next graph
        COLORS.append(COLORS.pop(0))

    plt.title(prefix)
    plt.legend(loc="best")
    plt.show()

if __name__ == "__main__":
    # Experiments on the same environment and plotted together
    prefix = sys.argv[1] if len(sys.argv) > 1 else ''
    window = int(sys.argv[2].split('=')[-1]) if len(sys.argv) > 2 else 1
    visualize(prefix, window)
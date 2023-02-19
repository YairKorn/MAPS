import os, csv, numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

PATH = os.path.join(os.getcwd(), "src", "classic", "analysis", "results")
DATA_FILE = "Analysis_Summary_2023-02-19-10:08:10.csv"
O_PATH = os.path.join(os.getcwd(), "src", "classic", "analysis")
GRAPHS = {'ORIGIN': 0, 'SURV': 1, 'OPTIMIZED': 2, 'LEARNING': 3}
ALPHAS = {'alpha=0.1': 0, 'alpha=0.5': 1, 'alpha=1.0': 2}

# Get the data
data = {}

with open(os.path.join(PATH, DATA_FILE), 'r') as f:
    raw_data = list(csv.reader(f))
    pass

    for raw in raw_data:
        env, alpha = raw[0], raw[1]
        if env not in data:
            data[env] = [0] * (2 * len(GRAPHS) * len(ALPHAS))
        
        for i, val in enumerate(raw[2:]):
            data[env][2 * len(GRAPHS) * ALPHAS[alpha] + i] = val

data_dir = os.path.join(PATH, "Graphs_" + datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
os.mkdir(data_dir)

# set width of bar
for i, k in enumerate(data.keys()):
        print(f"Processing {i}/{len(data.keys())}:\tEnv={k}")
        barWidth = 0.18
        fig = plt.subplots(figsize =(12, 8))

        br1 = np.arange(len(ALPHAS))
        br2 = [x + barWidth for x in br1]
        br3 = [x + barWidth for x in br2]
        br4 = [x + barWidth for x in br3]

        d = np.array(data[k], dtype=float)
        plt.bar(br1, d[::8], yerr=d[1::8], capsize=4, color ="salmon", width = barWidth,
                edgecolor ='red', label ='ORIGIN')
        plt.bar(br2, d[2::8], yerr=d[3::8], capsize=4, color ="burlywood", width = barWidth,
                edgecolor ='orange', label ='SURV')
        plt.bar(br3, d[4::8], yerr=d[5::8], capsize=4, color ="seagreen", width = barWidth,
                edgecolor ='green', label ='OPTIMIZED')
        plt.bar(br4, d[6::8], yerr=d[7::8], capsize=4, color ="royalblue", width = barWidth,
                edgecolor ='blue', label ='LEARNING')

        plt.xlabel('Optimization Const', fontweight ='bold', fontsize = 15)
        plt.ylabel('Optimization Value', fontweight ='bold', fontsize = 15)
        plt.xticks([r + barWidth for r in range(len(ALPHAS))],
                ['alpha=0.1', 'alpha=0.5', 'alpha=1.0'])
        plt.title('Optimization Results for ' + k)

        plt.legend()
        plt.savefig(os.path.join(data_dir, k))

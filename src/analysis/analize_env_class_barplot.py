import os, csv, yaml, numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

PATH = os.path.join(os.getcwd(), "src", "classic", "analysis", "results")
DATA_FILE = "Analysis_Summary_2023-02-19-10:08:10.csv"
O_PATH = os.path.join(os.getcwd(), "src", "classic", "analysis")
GRAPHS = {'ORIGIN': 0, 'OPTIMIZED': 1, 'LEARNING': 2}
ALPHAS = {'alpha=0.1': 0, 'alpha=0.5': 1, 'alpha=1.0': 2}
CLASSES = {
        "(0, 0)": "Connected & Flank Environment",
        "(0, 1)": "Connected & Divided Environment",
        "(1, 0)": "Scattered & Flank Environment",
        "(1, 1)": "Scattered & Divided Environment"
}

# Get the data
data = {}

with open(os.path.join(PATH, DATA_FILE), 'r') as f:
    raw_data = list(csv.reader(f))
    pass

    for raw in raw_data:
        env, alpha = raw[0], raw[1]
        if env not in data:
            data[env] = [0] * (2 * len(GRAPHS) * len(ALPHAS))
        
        del raw[4:6] # remove SURV
        for i, val in enumerate(raw[2:]):
            data[env][2 * len(GRAPHS) * ALPHAS[alpha] + i] = float(val)

# class_data = {(0, 0): [], (0, 1): [], (1, 0): [], (1, 1): []}
class_data = {}
with open(os.path.join(PATH, "Env_Classification.yaml"), 'r') as f:
        envs = yaml.safe_load(f)
        for env in envs:
                if envs[env] not in class_data:
                        class_data[envs[env]] = [np.zeros(2 * len(GRAPHS) * len(ALPHAS)), 0]
                class_data[envs[env]][0] += np.array(data[env])
                class_data[envs[env]][1] += 1

        for k in class_data.keys():
                class_data[k] = class_data[k][0] / class_data[k][1]


data_dir = os.path.join(PATH, "Class_" + datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
os.mkdir(data_dir)

# set width of bar
for i, k in enumerate(class_data.keys()):
        print(f"Processing {i+1}/{len(class_data.keys())}:\tEnv={k}")
        barWidth = 0.2
        fig = plt.subplots(figsize =(12, 8))

        br1 = np.arange(len(ALPHAS))
        br2 = [x + barWidth for x in br1]
        br3 = [x + barWidth for x in br2]

        d = np.array(class_data[k], dtype=float)
        plt.bar(br1, d[::6], yerr=d[1::6], capsize=4, color ="salmon", width = barWidth,
                edgecolor ='red', label ='ORIGIN')
        # plt.bar(br2, d[2::8], yerr=d[3::8], capsize=4, color ="burlywood", width = barWidth,
        #         edgecolor ='orange', label ='SURV')
        plt.bar(br2, d[2::6], yerr=d[3::6], capsize=4, color ="seagreen", width = barWidth,
                edgecolor ='green', label ='OPTIMIZED')
        plt.bar(br3, d[4::6], yerr=d[5::6], capsize=4, color ="royalblue", width = barWidth,
                edgecolor ='blue', label ='LEARNING')

        plt.xlabel('Optimization Const', fontweight ='bold', fontsize = 15)
        plt.ylabel('Optimization Value', fontweight ='bold', fontsize = 15)
        plt.xticks([r + barWidth for r in range(len(ALPHAS))],
                ['alpha=0.1', 'alpha=0.5', 'alpha=1.0'])
        plt.title('Optimization Results for ' + CLASSES[k])

        plt.legend()
        plt.savefig(os.path.join(data_dir, k))

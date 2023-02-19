import os, json, numpy as np
from tabulate import tabulate
from datetime import datetime
import csv

PATH = os.path.join(os.getcwd(), "src", "classic", "results")
HEADERS = ['Environment', 'Alpha', 'Graph', 'Cover', 'Time', 'Optim Mean', 'Optim STD']
O_PATH = os.path.join(os.getcwd(), "src", "classic", "analisys")

GRAPHS = {'ORIGIN': 0, 'SURV': 1, 'OPTIMIZED': 2}

if __name__ == '__main__':
    files = sorted(os.listdir(PATH))
    data = {}

    for file in files:
        env, alpha, graph, _ = file.split('#')
        fdata = json.load(open(os.path.join(PATH, file), 'r'))

        if env + '#' + alpha not in data:
            data[env + '#' + alpha] = [0] * 8

        data[env + '#' + alpha][2 * GRAPHS[graph]] = fdata["mean"]["optim_value"]
        data[env + '#' + alpha][2 * GRAPHS[graph] + 1] = fdata["std"]["optim_value"]
    
    # data_print = [[k, v] for k, v in data.items()]
    data_print = [k.split('#') + v for k, v in data.items()]
    with open(os.path.join(O_PATH, 'Analysis_Summary_' + datetime.now().strftime("%Y-%m-%d-%H:%M:%S") + '.csv'), 'w') as f:
        writer = csv.writer(f)
        # for d in data_print:
        writer.writerows(data_print)
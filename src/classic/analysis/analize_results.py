import os, json
from tabulate import tabulate
from datetime import datetime

PATH = os.path.join(os.getcwd(), "src", "classic", "results")
HEADERS = ['Environment', 'Alpha', 'Graph', 'Cover', 'Time', 'Optim Mean', 'Optim STD']
O_PATH = os.path.join(os.getcwd(), "src", "classic", "analisys")


if __name__ == '__main__':
    files = sorted(os.listdir(PATH))
    data = []

    for file in files:
        env, alpha, graph, _ = file.split('#')
        
        fdata = json.load(open(os.path.join(PATH, file), 'r'))
        values = [fdata["mean"]["coverage"], fdata["mean"]["time"], fdata["mean"]["optim_value"], fdata["std"]["optim_value"]]
        data.append([env, alpha, graph] + values)
    
    with open(os.path.join(O_PATH, 'Analysis_' + datetime.now().strftime("%Y-%m-%d-%H:%M:%S")), 'w') as f:
        f.write(tabulate(data, headers=HEADERS, numalign='center', tablefmt="github"))
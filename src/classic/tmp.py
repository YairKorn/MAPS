import matplotlib.pyplot as plt
import os, json

ENV = 'Connected_Areas3#alpha=0.1#OPTIMIZED#2023-02-15-18:16:17'

with open(os.path.join(os.getcwd(), 'src', 'classic', 'results', ENV + '.json'), 'r') as f:
    data = json.load(f)

    vals = data["optim_value"]
    plt.hist(vals, 20)
    plt.show()


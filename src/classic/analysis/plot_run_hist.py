import matplotlib.pyplot as plt
import os, json

ENV = 'Connected_Areas1#alpha=1.0#OPTIMIZED#2023-02-19-02:15:31'

with open(os.path.join(os.getcwd(), 'src', 'classic', 'results', ENV + '.json'), 'r') as f:
    data = json.load(f)

    vals = data["optim_value"]
    plt.hist(vals, 20)
    plt.show()


import subprocess

GRAPH = ['ORIGIN', 'SURV', 'OPTIMIZED'] # 
MAPS  = ['Random_Map11', 'Random_Map3'] # ['Connected_Areas1', 'Connected_Areas2', 'Connected_Areas3', 'Random_Map6', 'Random_Map9', 'Random_Map11']

def main():
    for map in MAPS:
        for graph in GRAPH:
            for alpha in [0.1, 0.5, 1.0]:
                run(map, graph, alpha)

                            
def run(map, graph, alpha):
    print(f"Run in {map} with graph={graph} and optim_alpha={alpha}")
    subprocess.run(['python', 'src/classic/mrac.py', map, graph, str(alpha)])


if __name__ == '__main__':
    main()
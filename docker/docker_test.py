import os, subprocess

ENV = 'het_adv_coverage'
ALGS = ['dcg'] # ['ppo', 'dql_tabular']
E_FACTOR = 0.6
REPEATS  = 1

args = {
    'save_model': True,

    'duelling': False
}

def main(map, t, alpha):
    for _ in range(REPEATS):
        for alg in ALGS:
            args['epsilon_anneal_time'] = int(E_FACTOR * t)
            args['env_args.reward_time'] = -alpha
            run(alg, map, t, args)

                            
def run(alg, map, t, args):
    args_str = ' '.join([f'{k}={v}' for k, v in args.items()])
    print(f"Run {alg} in {map} for {t} steps with {args_str}")

    subprocess.run(['python3', 'src/main.py', f'--config={alg}', f'--env-config={ENV}', 'with' \
        , f'env_args.map={map}', f't_max={t}'] + [f'{k}={v}' for k, v in args.items()])

# Run docker in using the next command:
# docker run -e map=[map] -e time=[time] -e alpha=[alpha] image
if __name__ == '__main__':
    run_vars = os.environ
    if (run_vars['map'] == "") or (run_vars['time'] == 0):
        raise "Missing arguments!"

    main(run_vars['map'], int(run_vars['time']), float(run_vars['alpha']))
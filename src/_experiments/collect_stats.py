import subprocess
import os
import pandas as pd

os.chdir(os.path.join(os.getcwd(), "..", ".."))
PATH = os.path.join(os.getcwd(), "results", "sacred")
ENV = 'adv_coverage'
MAPS = ['Random_Map3']

args = {
    'save_model': True,
    'obs_agent_id': True,
}


def main():
    all_maps = os.listdir(PATH)
    homo_maps = filter(lambda x: x[0] != "H" and x[0] != "_", all_maps)
    hetero_maps = filter(lambda x: x[0] == "H", all_maps)

    for test_map in homo_maps:
        per_group_runs = extract_runs(test_map)
        for run_row in per_group_runs.iterrows():
            extract_run_arguments(run_row, test_map)
            run('dcg', "adv_coverage", test_map, args)

    # for test_map in hetero_maps:
    #     per_group_runs = extract_runs(test_map)
    #     for run_row in per_group_runs.iterrows():
    #         extract_run_arguments(run_row, test_map)
    #         run('dcg', "het_adv_coverage", test_map, args)


def extract_run_arguments(run_row, test_map):
    alpha = float(run_row[1]["alpha"].split("=")[-1])
    run_dir_name = "#".join(run_row[1])
    args["env_args.reward_time"] = - alpha
    args["checkpoint_path"] = os.path.join(PATH, test_map, run_dir_name, "model")


def extract_runs(test_map):
    map_dir = os.path.join(PATH, test_map)
    map_runs = os.listdir(map_dir)
    map_runs = map(lambda x: x.split("#"), map_runs)
    map_runs_df = pd.DataFrame(map(lambda x: [x[0], x[1], "#".join(x[2:])], filter(lambda x: len(x) > 2, map_runs)),
                               columns=["alg", "alpha", "suffix"])
    per_group_runs = map_runs_df.sort_values("suffix").groupby("alpha").tail(1)
    return per_group_runs


def run(alg, env, map, args):
    args_str = ' '.join([f'{k}={v}' for k, v in args.items()])
    print(f"Run {alg} in {map} with {args_str}")

    try:
        subprocess.run(['python', 'src/main.py', f'--config={alg}', f'--env-config={env}', 'with',
                        f'env_args.map={map}', "-d"] + [f'{k}={v}' for k, v in args.items()])
    except Exception as e:
        print(f"Failed {e}")


if __name__ == '__main__':
    main()

import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu as mnw

RESULTS_PATH = os.path.join(os.getcwd(), "..", "results")
LEARNING_PATH = "/results/results_raw"
OUTPUT_PATH = os.path.join(os.getcwd(), "..", "analysis")


def calc_mnw(results):
    try:
        origin_f = results.query("algorithm == 'ORIGIN'").iloc[0]
        origin_json = json.load(open(os.path.join(RESULTS_PATH, "#".join(origin_f))))
        origin_results = np.array(origin_json["optim_value"])

        optimized_f = results.query("algorithm == 'OPTIMIZED'").iloc[0]
        optimized_json = json.load(open(os.path.join(RESULTS_PATH, "#".join(optimized_f))))
        optimized_results = np.array(optimized_json["optim_value"])

        if results.query("algorithm == 'LEARNING'").size > 0:
            learning_f = results.query("algorithm == 'LEARNING'").iloc[0]
            learning_json = json.load(open(os.path.join(LEARNING_PATH,
                                                        learning_f["map"] + "#" + learning_f["alpha"] + ".json")))
            learning_results = np.array(learning_json["optim_value"])
    except Exception:
        return None

    if results.query("algorithm == 'LEARNING'").size > 0:
        ret_df = pd.Series({
            "ORIGIN->OPTIMIZED": mnw(origin_results, optimized_results, alternative="two-sided")[1],
            "ORIGIN->LEARNING": mnw(origin_results, learning_results, alternative="two-sided")[1],
            "OPTIMIZED->LEARNING": mnw(optimized_results, learning_results, alternative="two-sided")[1],
        })
    else:
        ret_df = pd.Series({
            "ORIGIN->OPTIMIZED": mnw(origin_results, optimized_results, alternative="less")[1],
            "ORIGIN->LEARNING": None,
            "OPTIMIZED->LEARNING": None,
        })
    return ret_df


result_files = os.listdir(RESULTS_PATH)
learning_files = os.listdir(LEARNING_PATH)

# Heuristic results
results_df = pd.DataFrame([x.split("#") for x in result_files], columns=["map", "alpha", "algorithm", "suffix"])
results_df = results_df.loc[results_df.algorithm != "SURV", :]

# Learning results
learning_df = pd.DataFrame([x.replace(".json", "").split("#") for x in learning_files],
                           columns=["map", "alpha"])
learning_df.loc[:, "algorithm"] = "LEARNING"
learning_df.loc[:, "suffix"] = ".json"

all_results_df = pd.concat([results_df, learning_df])
results_mnw = all_results_df.groupby(["map", "alpha"]).apply(calc_mnw)

results_mnw.to_csv(os.path.join(OUTPUT_PATH, 'MNW_Analysis_' + datetime.now().strftime("%Y-%m-%d-%H:%M:%S") + '.csv'))

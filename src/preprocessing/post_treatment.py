import argparse
from train.utils import write_to_csv
import numpy as np
import os
import pandas as pd

parser = argparse.ArgumentParser()
# data parameters:
parser.add_argument("-folder_path", type=str,
                    default="../../output/RNN_weather/RNN_h32_ep15_bs64_maxsamples20000/20210417-080320/observations_samples1_seqlen25_sigmainit0.1_sigmah0.1_sigmay0.1/50_runs",
                    help="path for uploading the results")

args = parser.parse_args()

folder_path = args.folder_path

folder_path = "../../output/RNN_weather/RNN_h32_ep15_bs64_maxsamples20000/20210417-080320/observations_samples1_seqlen25_sigmainit0.1_sigmah0.1_sigmay0.1/50_runs"

dirs = [f.path for f in os.scandir(folder_path) if f.is_dir()]

path = "../../output/RNN_weather/RNN_h32_ep15_bs64_maxsamples20000/20210417-080320/observations_samples1_seqlen25_sigmainit0.1_sigmah0.1_sigmay0.1/experiments/results_1runs_4J_10particles_10pms-part.csv"

backward_results = []
pms_results = []

for dir in dirs:
    for file in os.listdir(dir):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(dir, file), index_col=0)
            backward_res = df.loc["backward_is_mean", "0"]
            pms_res = df.loc["pms_mean", "0"]
            backward_results.append(backward_res)
            pms_results.append(pms_res)

aggregated_results = {"backward_mean": np.mean(backward_results), "backward_var": np.var(backward_results), "backward_results": backward_results,
                      "pms_mean": np.mean(pms_results), "pms_var": np.var(pms_results), "pms_results": pms_results}

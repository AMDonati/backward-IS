import argparse
import numpy as np
import os
import pandas as pd

parser = argparse.ArgumentParser()
# data parameters:
parser.add_argument("-folder_path", type=str,
                    required=True,
                    help="path for uploading the results")

args = parser.parse_args()

folder_path = args.folder_path
dirs = [f.path for f in os.scandir(folder_path) if f.is_dir()]

backward_results = []
pms_results = []

for dir in dirs:
    for file in os.listdir(dir):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(dir, file), index_col=0)
            backward_res = float(df.loc["backward_is_mean", "0"])
            pms_res = float(df.loc["pms_mean", "0"])
            backward_results.append(backward_res)
            pms_results.append(pms_res)

aggregated_results = {"backward_mean": np.mean(backward_results), "backward_var": np.var(backward_results), "backward_results": backward_results,
                      "pms_mean": np.mean(pms_results), "pms_var": np.var(pms_results), "pms_results": pms_results}

pd.DataFrame(data=aggregated_results).to_csv(os.path.join(folder_path, "aggregated_results.csv"))

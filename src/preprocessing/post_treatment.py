import argparse
import numpy as np
import os
import pandas as pd
from smc.plots import plot_online_estimation_mse_aggregated, plot_estimation_Xk
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

for i, dir in enumerate(dirs):
    df = pd.read_csv(os.path.join(dir, "results.csv"), index_col=0)
    df.loc["pms_all_seq", :] = df.loc["pms_all_seq", :].astype(dtype=float)
    df.loc["backward_all_seq", :] = df.loc["backward_all_seq", :].astype(dtype=float)
    backward_seq_0 = np.load(os.path.join(dir, "backward_is", "backward_by_seq_0.npy"))
    backward_seq_mean = np.load(os.path.join(dir, "backward_is", "backward_by_seq_mean.npy"))
    pms_seq_0 = np.load(os.path.join(dir, "backward_is", "pms_by_seq_0.npy"))
    pms_seq_mean = np.load(os.path.join(dir, "backward_is", "pms_by_seq_mean.npy"))
    if i == 0:
        final_df = df.loc[["pms_all_seq", "backward_all_seq"], :]
        mean_backward_seq_0 = backward_seq_0
        mean_backward_seq_mean = backward_seq_mean
        mean_pms_seq_0 = pms_seq_0
        mean_pms_seq_mean = pms_seq_mean
    else:
        final_df.loc["pms_all_seq", :] += df.loc["pms_all_seq", :]
        final_df.loc["backward_all_seq", :] += df.loc["backward_all_seq", :]
        mean_backward_seq_0 += backward_seq_0
        mean_backward_seq_mean += backward_seq_mean
        mean_pms_seq_0 += pms_seq_0
        mean_pms_seq_mean += pms_seq_mean

final_df.loc["pms_all_seq", :] = final_df.loc["pms_all_seq", :] / len(dirs)
final_df.loc["backward_all_seq", :] = final_df.loc["backward_all_seq", :] / len(dirs)

mean_backward_seq_0 = mean_backward_seq_0 / len(dirs)
mean_backward_seq_mean = mean_backward_seq_mean / len(dirs)
mean_pms_seq_0 = mean_pms_seq_0 / len(dirs)
mean_pms_seq_mean = mean_pms_seq_mean / len(dirs)

pms_all_k = final_df.iloc[0, :-1]
backward_all_k = final_df.iloc[1, :-1]


out_folder = os.path.join(folder_path, "aggregated_results")
if not os.path.isdir(out_folder):
    os.makedirs(out_folder)
plot_online_estimation_mse_aggregated(pms_by_seq=mean_pms_seq_0, backward_by_seq=mean_backward_seq_0, out_folder=out_folder)
plot_estimation_Xk(pms_all_k=pms_all_k.values, backward_all_k=backward_all_k.values, out_folder=out_folder)

final_df.to_csv(os.path.join(out_folder, "aggregated_results.csv"))

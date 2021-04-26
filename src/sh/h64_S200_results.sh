#!/bin/bash
#SBATCH --job-name=H64-S100
#SBATCH --qos=qos_gpu-t4
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8g
#SBATCH --output=slurm_out_2/h64-S100-%j.out
#SBATCH --error=slurm_out_2/h64-S100-%j.err
#SBATCH --time=50:00:00


export TMPDIR=$JOBSCRATCH
module purge
module load  pytorch-gpu/py3/1.7.1
conda activate back-is

export PYTHONPATH=src:${PYTHONPATH}

FOLDER_PATH="output/RNN_weather/RNN_h64_ep15_bs64_maxsamples20000/20210416-225828/seq_len200/100_runs"

set -x
srun python -u src/preprocessing/post_treatment.py -folder_path $FOLDER_PATH
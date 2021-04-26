#!/bin/bash
#SBATCH --job-name=TEST-H64-S200
#SBATCH --qos=qos_gpu-t3
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8g
#SBATCH --output=slurm_out_2/TEST-h64-S200-%j.out
#SBATCH --error=slurm_out_2/TEST-h64-S200-%j.err
#SBATCH --time=5:00:00


export TMPDIR=$JOBSCRATCH
module purge
module load  pytorch-gpu/py3/1.7.1
conda activate back-is

export PYTHONPATH=src:${PYTHONPATH}

FOLDER_PATH="output/RNN_weather/RNN_h64_ep15_bs64_maxsamples20000/20210416-225828/seq_len200/100_runs"

set -x
srun python -u src/preprocessing/post_treatment.py -folder_path $FOLDER_PATH
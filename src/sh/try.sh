#!/bin/bash
#SBATCH --job-name=debug
#SBATCH --qos=qos_gpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8g
#SBATCH --output=slurm_out/debug-j.out
#SBATCH --error=slurm_out/debug-j.err
#SBATCH --time=10:00:00

export TMPDIR=$JOBSCRATCH
module purge
module load  pytorch-gpu/py3/1.7.1
conda activate back-is

export PYTHONPATH=src:${PYTHONPATH}

DATA_PATH="output/RNN_weather/RNN_h32_ep15_bs64_maxsamples20000/20210417-080320/observations_samples1_seqlen25_sigmainit0.1_sigmah0.1_sigmay0.1"
MODEL_PATH="output/RNN_weather/RNN_h32_ep15_bs64_maxsamples20000/20210417-080320/model.pt"
SIGMA_INIT=0.1
SIGMA_Y=0.1
SIGMA_H=0.1
NUM_PARTICLES=5
BACKWARD_SAMPLES=4
RUNS=2

set -x
srun python -u src/estimate.py -data_path $DATA_PATH -model_path $MODEL_PATH -num_particles $NUM_PARTICLES -backward_samples $BACKWARD_SAMPLES -sigma_init $SIGMA_INIT -sigma_y $SIGMA_Y -sigma_h $SIGMA_H -runs $RUNS
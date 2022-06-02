#!/bin/bash
#SBATCH --job-name=PMS-state-2000P
#SBATCH --qos=qos_gpu-t3
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --output=slurm_out/PMS-state-2000p-%j.out
#SBATCH --error=slurm_out/PMS-state-2000p-%j.err
#SBATCH --time=20:00:00
#SBATCH -A ktz@v100


export TMPDIR=$JOBSCRATCH
module purge
module load  pytorch-gpu/py3/1.7.1
conda activate back-is

export PYTHONPATH=src:${PYTHONPATH}

DATA_PATH="data/SV"
OUT_PATH="experiments"
NUM_PARTICLES=2000
BACKWARD_SAMPLES=16
ALGO="PMS"
N_ITER=50
INIT_PARAMS="random"
ESTIM="state"
SEQ_LEN=100

srun python -u src/train/EM_algo.py -data_path $DATA_PATH -out_path $OUT_PATH -num_particles $NUM_PARTICLES -backward_samples $BACKWARD_SAMPLES -algo $ALGO -n_iter $N_ITER -init_params $INIT_PARAMS -estim $ESTIM -seq_len $SEQ_LEN
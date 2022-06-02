#!/bin/bash
#SBATCH --job-name=PMS-params-1000P-randinit
#SBATCH --qos=qos_gpu-t4
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --output=slurm_out/PMS-params-1000p-randinit-%j.out
#SBATCH --error=slurm_out/PMS-params-1000p-randinit-%j.err
#SBATCH --time=100:00:00
#SBATCH -A ktz@v100


export TMPDIR=$JOBSCRATCH
module purge
module load  pytorch-gpu/py3/1.7.1
conda activate back-is

export PYTHONPATH=src:${PYTHONPATH}

DATA_PATH="data/SV"
OUT_PATH="experiments"
NUM_PARTICLES=1000
BACKWARD_SAMPLES=16
ALGO="PMS"
N_ITER=50
INIT_PARAMS="random"
ESTIM="parameter"
SEQ_LEN=100

srun python -u src/train/EM_algo.py -data_path $DATA_PATH -out_path $OUT_PATH -num_particles $NUM_PARTICLES -backward_samples $BACKWARD_SAMPLES -algo $ALGO -n_iter $N_ITER -init_params $INIT_PARAMS -estim $ESTIM -seq_len $SEQ_LEN
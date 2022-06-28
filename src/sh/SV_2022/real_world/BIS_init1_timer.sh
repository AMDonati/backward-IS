#!/bin/bash
#SBATCH --job-name=timer-RW-BIS-params1
#SBATCH --qos=qos_gpu-t3
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_out/RW-BIS-TIMER-%j.out
#SBATCH --error=slurm_out/RW-BIS-TIMER-%j.err
#SBATCH --time=20:00:00
#SBATCH -A ktz@v100


export TMPDIR=$JOBSCRATCH
module purge
module load  pytorch-gpu/py3/1.7.1
conda activate back-is

export PYTHONPATH=src:${PYTHONPATH}

DATA_PATH="data"
OUT_PATH="experiments_RW/timer"
NUM_PARTICLES=100
BACKWARD_SAMPLES=16
ALGO="BIS"
N_ITER=1
ALPHA=0.5
SIGMA=0.7
BETA=0.1

echo "number of particles = 100"
set -x
srun python -u src/train/EM_algo_realworld.py -data_path $DATA_PATH -out_path $OUT_PATH -num_particles $NUM_PARTICLES -backward_samples $BACKWARD_SAMPLES -algo $ALGO -n_iter $N_ITER -alpha $ALPHA -sigma $SIGMA -beta $BETA

NUM_PARTICLES=250

echo "----------------------------------------------------------------------------------------------------------"
echo "number of particles = 250"
set -x
srun python -u src/train/EM_algo_realworld.py -data_path $DATA_PATH -out_path $OUT_PATH -num_particles $NUM_PARTICLES -backward_samples $BACKWARD_SAMPLES -algo $ALGO -n_iter $N_ITER -alpha $ALPHA -sigma $SIGMA -beta $BETA

NUM_PARTICLES=500


echo "----------------------------------------------------------------------------------------------------------"
echo "number of particles = 500"

set -x
srun python -u src/train/EM_algo_realworld.py -data_path $DATA_PATH -out_path $OUT_PATH -num_particles $NUM_PARTICLES -backward_samples $BACKWARD_SAMPLES -algo $ALGO -n_iter $N_ITER -alpha $ALPHA -sigma $SIGMA -beta $BETA

ALGO="PMS"

echo "---------------------------------------------------PMS---------------------------------------------------------------------------"
echo "number of particles = 5000"

NUM_PARTICLES=5000

set -x
srun python -u src/train/EM_algo_realworld.py -data_path $DATA_PATH -out_path $OUT_PATH -num_particles $NUM_PARTICLES -backward_samples $BACKWARD_SAMPLES -algo $ALGO -n_iter $N_ITER -alpha $ALPHA -sigma $SIGMA -beta $BETA

NUM_PARTICLES=7500

echo "----------------------------------------------------------------------------------------------------------"
echo "number of particles = 5000"

set -x
srun python -u src/train/EM_algo_realworld.py -data_path $DATA_PATH -out_path $OUT_PATH -num_particles $NUM_PARTICLES -backward_samples $BACKWARD_SAMPLES -algo $ALGO -n_iter $N_ITER -alpha $ALPHA -sigma $SIGMA -beta $BETA

NUM_PARTICLES=10000

echo "----------------------------------------------------------------------------------------------------------"
echo "number of particles = 10000"

set -x
srun python -u src/train/EM_algo_realworld.py -data_path $DATA_PATH -out_path $OUT_PATH -num_particles $NUM_PARTICLES -backward_samples $BACKWARD_SAMPLES -algo $ALGO -n_iter $N_ITER -alpha $ALPHA -sigma $SIGMA -beta $BETA


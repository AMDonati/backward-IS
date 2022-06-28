#!/bin/bash
#SBATCH --job-name=RW-PMS-params3
#SBATCH --qos=qos_gpu-t3
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --array=1-10
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_out/RW-PMS-params3-%j.out
#SBATCH --error=slurm_out/RW-PMS-params3-%j.err
#SBATCH --time=20:00:00
#SBATCH -A ktz@v100


export TMPDIR=$JOBSCRATCH
module purge
module load  pytorch-gpu/py3/1.7.1
conda activate back-is

export PYTHONPATH=src:${PYTHONPATH}

DATA_PATH="data"
NUM_PARTICLES=10000
BACKWARD_SAMPLES=16
ALGO="PMS"
N_ITER=40
ALPHA=0.5
SIGMA=0.7
BETA=0.1

set -x
echo "now processing task id:: " ${SLURM_ARRAY_TASK_ID}
OUT_PATH=experiments_RW/exp_param/${SLURM_ARRAY_TASK_ID}
srun python -u src/train/EM_algo_realworld.py -data_path $DATA_PATH -out_path ${OUT_PATH} -num_particles $NUM_PARTICLES -backward_samples $BACKWARD_SAMPLES -algo $ALGO -n_iter $N_ITER -alpha $ALPHA -sigma $SIGMA -beta $BETA
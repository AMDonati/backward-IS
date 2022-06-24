#!/bin/bash
#SBATCH --job-name=RW-BIS-params1
#SBATCH --qos=qos_gpu-t3
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --array=1-5
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_out/RW-BIS-params1-%j.out
#SBATCH --error=slurm_out/RW-BIS-params1-%j.err
#SBATCH --time=20:00:00
#SBATCH -A ktz@v100


export TMPDIR=$JOBSCRATCH
module purge
module load  pytorch-gpu/py3/1.7.1
conda activate back-is

export PYTHONPATH=src:${PYTHONPATH}

DATA_PATH="data"
NUM_PARTICLES=100
BACKWARD_SAMPLES=16
ALGO="BIS"
N_ITER=30
ALPHA=0.91
SIGMA=1.0
BETA=0.5

set -x
echo "now processing task id:: " ${SLURM_ARRAY_TASK_ID}
OUT_PATH=experiments_RW/${SLURM_ARRAY_TASK_ID}
srun python -u src/train/EM_algo_realworld.py -data_path $DATA_PATH -out_path ${OUT_PATH} -num_particles $NUM_PARTICLES -backward_samples $BACKWARD_SAMPLES -algo $ALGO -n_iter $N_ITER -alpha $ALPHA -sigma $SIGMA -beta $BETA
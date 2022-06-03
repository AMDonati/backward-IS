#!/bin/bash
#SBATCH --job-name=PMS-random1
#SBATCH --qos=qos_gpu-t3
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --array=1-10
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_out/PMS10seeds-params-100p-randinit1-%j.out
#SBATCH --error=slurm_out/PMS10seeds-params-100p-randinit1-%j.err
#SBATCH --time=20:00:00
#SBATCH -A ktz@v100


export TMPDIR=$JOBSCRATCH
module purge
module load  pytorch-gpu/py3/1.7.1
conda activate back-is

export PYTHONPATH=src:${PYTHONPATH}

NUM_PARTICLES=25000
BACKWARD_SAMPLES=16
ALGO="PMS"
N_ITER=50
INIT_PARAMS="random1"
ESTIM="parameter"
SEQ_LEN=500

set -x
echo "now processing task id:: " ${SLURM_ARRAY_TASK_ID}
OUT_PATH=experiments/1Oseeds_2/${SLURM_ARRAY_TASK_ID}
srun python -u src/train/EM_algo.py -out_path ${OUT_PATH} -num_particles $NUM_PARTICLES -backward_samples $BACKWARD_SAMPLES -algo $ALGO -n_iter $N_ITER -init_params $INIT_PARAMS -estim $ESTIM -seq_len $SEQ_LEN
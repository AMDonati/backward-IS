#!/bin/bash
#SBATCH --job-name=time-backward
#SBATCH --qos=qos_gpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8g
#SBATCH --output=slurm_out/time-%j.out
#SBATCH --error=slurm_out/time-%j.err
#SBATCH --time=10:00:00

export TMPDIR=$JOBSCRATCH
module purge
module load  pytorch-gpu/py3/1.7.1
conda activate back-is

export PYTHONPATH=src:${PYTHONPATH}

DATA_PATH="output/RNN_weather/RNN_h32_ep15_bs64_maxsamples20000/20210417-080320/observations_samples1_seqlen101_sigmainit0.1_sigmah0.1_sigmay0.1"
MODEL_PATH="output/RNN_weather/RNN_h32_ep15_bs64_maxsamples20000/20210417-080320/model.pt"
OUT_PATH="time_backward"
SIGMA_INIT=0.1
SIGMA_Y=0.1
SIGMA_H=0.1
NUM_PARTICLES=1000
BACKWARD_SAMPLES=32
RUNS=1
PMS=1
PARTICLES_PMS=10000

set -x
PARTICLES_PMS=4000

srun python -u src/estimate.py -data_path $DATA_PATH -model_path $MODEL_PATH -out_path $OUT_PATH -num_particles $NUM_PARTICLES -backward_samples $BACKWARD_SAMPLES -sigma_init $SIGMA_INIT -sigma_y $SIGMA_Y -sigma_h $SIGMA_H -runs $RUNS -pms $PMS -particles_pms $PARTICLES_PMS

PARTICLES_PMS=3000

srun python -u src/estimate.py -data_path $DATA_PATH -model_path $MODEL_PATH -out_path $OUT_PATH -num_particles $NUM_PARTICLES -backward_samples $BACKWARD_SAMPLES -sigma_init $SIGMA_INIT -sigma_y $SIGMA_Y -sigma_h $SIGMA_H -runs $RUNS -pms $PMS -particles_pms $PARTICLES_PMS

PARTICLES_PMS=2000

srun python -u src/estimate.py -data_path $DATA_PATH -model_path $MODEL_PATH -out_path $OUT_PATH -num_particles $NUM_PARTICLES -backward_samples $BACKWARD_SAMPLES -sigma_init $SIGMA_INIT -sigma_y $SIGMA_Y -sigma_h $SIGMA_H -runs $RUNS -pms $PMS -particles_pms $PARTICLES_PMS


PARTICLES_PMS=1000

srun python -u src/estimate.py -data_path $DATA_PATH -model_path $MODEL_PATH -out_path $OUT_PATH -num_particles $NUM_PARTICLES -backward_samples $BACKWARD_SAMPLES -sigma_init $SIGMA_INIT -sigma_y $SIGMA_Y -sigma_h $SIGMA_H -runs $RUNS -pms $PMS -particles_pms $PARTICLES_PMS
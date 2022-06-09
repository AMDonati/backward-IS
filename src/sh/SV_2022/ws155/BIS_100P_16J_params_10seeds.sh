#!/bin/bash
NUM_PARTICLES=100
BACKWARD_SAMPLES=16
ALGO="BIS"
N_ITER=50
INIT_PARAMS="random1"
ESTIM="parameter"
SEQ_LEN=1000

OUT_PATH="experiments/1Oseeds/1"
python src/train/EM_algo.py -out_path $OUT_PATH -num_particles $NUM_PARTICLES -backward_samples $BACKWARD_SAMPLES -algo $ALGO -n_iter $N_ITER -init_params $INIT_PARAMS -estim $ESTIM -seq_len $SEQ_LEN

OUT_PATH="experiments/1Oseeds/2"
python src/train/EM_algo.py -out_path $OUT_PATH -num_particles $NUM_PARTICLES -backward_samples $BACKWARD_SAMPLES -algo $ALGO -n_iter $N_ITER -init_params $INIT_PARAMS -estim $ESTIM -seq_len $SEQ_LEN

OUT_PATH="experiments/1Oseeds/3"
python src/train/EM_algo.py -out_path $OUT_PATH -num_particles $NUM_PARTICLES -backward_samples $BACKWARD_SAMPLES -algo $ALGO -n_iter $N_ITER -init_params $INIT_PARAMS -estim $ESTIM -seq_len $SEQ_LEN

OUT_PATH="experiments/1Oseeds/4"
python src/train/EM_algo.py -out_path $OUT_PATH -num_particles $NUM_PARTICLES -backward_samples $BACKWARD_SAMPLES -algo $ALGO -n_iter $N_ITER -init_params $INIT_PARAMS -estim $ESTIM -seq_len $SEQ_LEN

OUT_PATH="experiments/1Oseeds/5"
python src/train/EM_algo.py -out_path $OUT_PATH -num_particles $NUM_PARTICLES -backward_samples $BACKWARD_SAMPLES -algo $ALGO -n_iter $N_ITER -init_params $INIT_PARAMS -estim $ESTIM -seq_len $SEQ_LEN

OUT_PATH="experiments/1Oseeds/6"
python src/train/EM_algo.py -out_path $OUT_PATH -num_particles $NUM_PARTICLES -backward_samples $BACKWARD_SAMPLES -algo $ALGO -n_iter $N_ITER -init_params $INIT_PARAMS -estim $ESTIM -seq_len $SEQ_LEN

OUT_PATH="experiments/1Oseeds/7"
python src/train/EM_algo.py -out_path $OUT_PATH -num_particles $NUM_PARTICLES -backward_samples $BACKWARD_SAMPLES -algo $ALGO -n_iter $N_ITER -init_params $INIT_PARAMS -estim $ESTIM -seq_len $SEQ_LEN

OUT_PATH="experiments/1Oseeds/8"
python src/train/EM_algo.py -out_path $OUT_PATH -num_particles $NUM_PARTICLES -backward_samples $BACKWARD_SAMPLES -algo $ALGO -n_iter $N_ITER -init_params $INIT_PARAMS -estim $ESTIM -seq_len $SEQ_LEN

OUT_PATH="experiments/1Oseeds/9"
python src/train/EM_algo.py -out_path $OUT_PATH -num_particles $NUM_PARTICLES -backward_samples $BACKWARD_SAMPLES -algo $ALGO -n_iter $N_ITER -init_params $INIT_PARAMS -estim $ESTIM -seq_len $SEQ_LEN

OUT_PATH="experiments/1Oseeds/10"
python src/train/EM_algo.py -out_path $OUT_PATH -num_particles $NUM_PARTICLES -backward_samples $BACKWARD_SAMPLES -algo $ALGO -n_iter $N_ITER -init_params $INIT_PARAMS -estim $ESTIM -seq_len $SEQ_LEN
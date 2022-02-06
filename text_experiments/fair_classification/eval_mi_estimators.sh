#!/usr/bin/env bash

export SAVE_STEPS=100
export FILTER=-1
export EPOCHS=3
export LAYERS=2
export SEED=49
export BATCH_SIZE=32
export LR=0.0001
export RENY_TRAINING=2
export HIDDEN_SIZE=128
export NOISE_P=0.2
export CONTENT_DIM=128
export DEC_HIDDEN_DIM=128

for checkpoints_number in "90000" "80000" "30000" "40000" "50000" "60000" "70000" "20000" "10000"; do
  for MI_ESTIMATOR in "NWJ" "InfoNCE" "L1OutUB" "CLUB"; do
    for STYLE_DIM in 1; do
      for MODEL in 'style_emb'; do

        for MUL in 0.001 0.01 0.1 1 10 100 1000; do

          export MODEL_PATH="style_emb_lambda_${MUL}_${MI_ESTIMATOR}"
          echo $MODEL_PATH
          echo $checkpoints_number
          export suffix=${MODEL_PATH}_${checkpoints_number}
          export MODEL_PATH=${MODEL_PATH}/checkpoint-${checkpoints_number}
          export OUTPUT_DIR="${MODEL_PATH}_"

          sbatch --job-name=${MODEL}_${MUL}_${SUFFIX} \
            --partition=gpu_p1 \
            --gres=gpu:1 \
            --no-requeue \
            --cpus-per-task=10 \
            --hint=nomultithread \
            --time=1:00:00 \
            --output=jobinfo/${MODEL}_${MUL}_${SUFFIX}_%j.out \
            --error=jobinfo/${MODEL}_${MUL}_${SUFFIX}_%j.err \
            --qos=qos_gpu-t4 \
            --wrap="module purge; module load pytorch-gpu/py3/1.1 ;  python evaluate_for_style_transfert.py --use_complex_classifier --use_gender  --save_step=$SAVE_STEPS  --suffix=$suffix  --num_train_epochs=$EPOCHS  --do_train_classifer  --do_test_transfer  --do_test_reconstruction  --do_train_classifer  --do_eval  --output_dir=$OUTPUT_DIR  --batch_size=$BATCH_SIZE  --filter=-1  --model=$MODEL  --model_path_to_load=$MODEL_PATH  --saving_result_file=evaluate_for_style_transfert_${checkpoints_number}.txt "
        done

      done

    done
  done
done

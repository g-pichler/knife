#!/usr/bin/env bash

export SAVE_STEPS=1000
export FILTER=-1
export EPOCHS=30
export EVAL_STEPS=1000
export LAYERS=2
export SEED=49
export BATCH_SIZE=128
export LR=0.0001
export RENY_TRAINING=2
export HIDDEN_SIZE=128
export NOISE_P=0.2
export CONTENT_DIM=128
export DEC_HIDDEN_DIM=128
for MI_ESTIMATOR in "NWJ" "InfoNCE" "L1OutUB" "CLUB" "DOE"; do
  for STYLE_DIM in 1; do
    for MODEL in 'special_baseline'; do

      for MUL in 0.001 0.01 0.1 1 10 100 1000; do

        export ALPHA=1.3

        export SUFFIX=classif_newmention_mi_${MI_ESTIMATOR}
        export OUTPUT_DIR=${MODEL}_lambda_${MUL}_${SUFFIX}

        sbatch --job-name=${MODEL}_${MUL}_${SUFFIX} \
          --gres=gpu:1 \
          --no-requeue \
          --cpus-per-task=10 \
          --hint=nomultithread \
          --time=10:00:00 \
          --output=jobinfo/${MODEL}_${MUL}_${SUFFIX}_%j.out \
          --error=jobinfo/${MODEL}_${MUL}_${SUFFIX}_%j.err \
          --qos=qos_gpu-t3 \
          --wrap="module purge; module load pytorch-gpu/py3/1.7.0 ;  python train.py --use_mention --mi_estimator_name=${MI_ESTIMATOR} --classif  --use_complex_classifier --use_gender   --no_reny --noise_p=$NOISE_P --content_dim=$CONTENT_DIM --hidden_dim=$HIDDEN_SIZE --dec_hidden_dim=$DEC_HIDDEN_DIM --not_use_ema --mul_mi=$MUL --reny_training=$RENY_TRAINING --style_dim=$STYLE_DIM --batch_size=$BATCH_SIZE --seed=$SEED --save_step=$SAVE_STEPS --learning_rate=$LR --output_dir=$OUTPUT_DIR --num_train_epochs=$EPOCHS --eval_step=$EVAL_STEPS --number_of_layers=$LAYERS --model=$MODEL --filter=$FILTER --alpha=$ALPHA"
      done

    done

    for STYLE_DIM in 1; do
      for MODEL in 'special_baseline'; do

        for MUL in 0.001 0.01 0.1 1 10 100 1000; do

          export ALPHA=1.3

          export SUFFIX=classif_newsentiment_mi_${MI_ESTIMATOR}
          export OUTPUT_DIR=${MODEL}_lambda_${MUL}_${SUFFIX}

          sbatch --job-name=${MODEL}_${MUL}_${SUFFIX} \
            --gres=gpu:1 \
            --no-requeue \
            --cpus-per-task=10 \
            --hint=nomultithread \
            --time=10:00:00 \
            --output=jobinfo/${MODEL}_${MUL}_${SUFFIX}_%j.out \
            --error=jobinfo/${MODEL}_${MUL}_${SUFFIX}_%j.err \
            --qos=qos_gpu-t3 \
            --wrap="module purge; module load pytorch-gpu/py3/1.7.0 ;  python train.py  --mi_estimator_name=${MI_ESTIMATOR} --classif  --use_complex_classifier   --use_gender   --no_reny --noise_p=$NOISE_P --content_dim=$CONTENT_DIM --hidden_dim=$HIDDEN_SIZE --dec_hidden_dim=$DEC_HIDDEN_DIM --not_use_ema --mul_mi=$MUL --reny_training=$RENY_TRAINING --style_dim=$STYLE_DIM --batch_size=$BATCH_SIZE --seed=$SEED --save_step=$SAVE_STEPS --learning_rate=$LR --output_dir=$OUTPUT_DIR --num_train_epochs=$EPOCHS --eval_step=$EVAL_STEPS --number_of_layers=$LAYERS --model=$MODEL --filter=$FILTER --alpha=$ALPHA"

        done

      done

    done
  done
done

#!/usr/bin/env bash
SEED=8998
EPOCH=10

for TASK in "mrpc" "sts-b"; do
  for SEED in 1 2 3 4 5 6 7 8 9; do
    sbatch --job-name=bert_%j \
      --account=${IDRPROJ}@gpu \
      --gres=gpu:1 \
      --no-requeue \
      --cpus-per-task=10 \
      --hint=nomultithread \
      --time=1:00:00 \
      --output=jobinfo/${MODEL}_${MUL}_${SUFFIX}_%j.out \
      --error=jobinfo/${MODEL}_${MUL}_${SUFFIX}_%j.err \
      --qos=qos_gpu-t4 \
      --wrap="module purge; module load pytorch-gpu/py3/1.2.0 ;  python train.py --model_name_or_path /gpfswork/rech/qsq/uwi62ct/transformers_models/bert-base-uncased  --output_dir out_dir_baseline_${SEED} --task_name ${TASK} --model_name_or_path=/gpfswork/rech/qsq/uwi62ct/transformers_models/bert-base-uncased --overwrite_cache --do_eval --max_seq_length 128 --num_train_epochs ${EPOCH} --overwrite_output_dir --outputfile results_${TASK}/results_baseline_${SEED}.csv --do_lower_case --eval_types train dev test --learning_rate 2e-5 --per_gpu_train_batch_size 32 --do_train --model_type=bert --evaluate_after_each_epoch --seed ${SEED}"
    for IB_DIM in 144 192 288 384; do
      for BETA in 0.001 0.0001 0.00001; do

        sbatch --job-name=bert_%j \
          --account=${IDRPROJ}@gpu \
          --gres=gpu:1 \
          --no-requeue \
          --cpus-per-task=10 \
          --hint=nomultithread \
          --time=1:00:00 \
          --output=jobinfo/bert_%j.out \
          --error=jobinfo/bert_%j.err \
          --qos=qos_gpu-t4 \
          --wrap="module purge; module load pytorch-gpu/py3/1.2.0 ; python train.py --per_gpu_train_batch_size 32 --model_name_or_path /gpfswork/rech/qsq/uwi62ct/transformers_models/bert-base-uncased  --output_dir out_dir_ib_baseline_${SEED}_${IB_DIM}_${BETA}_${TASK} --task_name ${TASK} --model_name_or_path=/gpfswork/rech/qsq/uwi62ct/transformers_models/bert-base-uncased --overwrite_cache --do_eval --max_seq_length 128 --num_train_epochs ${EPOCH} --overwrite_output_dir --outputfile results_${TASK}/results_ibbaseline_${SEED}_${IB_DIM}_${BETA}.csv --do_lower_case --ib_dim ${IB_DIM} --beta ${BETA} --ib --learning_rate 2e-5 --do_train --model_type=bert --eval_types dev train test --kl_annealing linear --evaluate_after_each_epoch --seed ${SEED}"

        for ESTIMATOR in 'CLUBSample' 'MINE' 'L1OutUB' 'NWJ' 'InfoNCE' 'KERNEL_E' 'KERNEL_A'; do

          sbatch --job-name=bert_%j \
            --gres=gpu:1 \
            --account=${IDRPROJ}@gpu \
            --no-requeue \
            --cpus-per-task=10 \
            --hint=nomultithread \
            --time=1:00:00 \
            --output=jobinfo/bert_%j.out \
            --error=jobinfo/bert_%j.err \
            --qos=qos_gpu-t4 \
            --wrap="module purge; module load pytorch-gpu/py3/1.2.0 ;  python train.py --per_gpu_train_batch_size 32 --model_name_or_path /gpfswork/rech/qsq/uwi62ct/transformers_models/bert-base-uncased --output_dir out_dir_${ESTIMATOR}_${SEED}_${IB_DIM}_${BETA}_${TASK} --task_name ${TASK} --model_name_or_path=/gpfswork/rech/qsq/uwi62ct/transformers_models/bert-base-uncased --overwrite_cache --do_eval --max_seq_length 128 --num_train_epochs ${EPOCH} --overwrite_output_dir --outputfile results_${TASK}/results_${ESTIMATOR}_${SEED}_${IB_DIM}_${BETA}.csv --do_lower_case --ib_dim ${IB_DIM} --beta ${BETA} --ib --learning_rate 2e-5 --do_train --model_type=bert --eval_types dev train test --kl_annealing linear --evaluate_after_each_epoch --seed ${SEED} --use_mi_estimation --name_mi_estimator=${ESTIMATOR}"
        done
      done
    done
  done
done

#!/usr/bin/env bash
export NCCL_TIMEOUT=2400
CUDA_VISIBLE_DEVICES=1,2 accelerate launch --num_processes=2 run_pseudo_labelling.py \
  --model_name_or_path "openai/whisper-small" \
  --dataset_name "mozilla-foundation/common_voice_16_1" \
  --dataset_config_name "zh-CN" \
  --dataset_split_name "train+validation+test" \
  --text_column_name "sentence" \
  --id_column_name "path" \
  --output_dir "./common_voice_16_1_zh-CN_pseudo_labelled" \
  --wandb_project "distil-whisper-labelling" \
  --per_device_eval_batch_size 64 \
  --dtype "bfloat16" \
  --attn_implementation "flash_attention_2" \
  --logging_steps 500 \
  --max_label_length 256 \
  --concatenate_audio True \
  --preprocessing_batch_size 250 \
  --preprocessing_num_workers 6 \
  --dataloader_num_workers 6 \
  --report_to "wandb" \
  --language "chinese" \
  --task "transcribe" \
  --return_timestamps \
  --streaming False \
  --generation_num_beams 1 \
  --push_to_hub False
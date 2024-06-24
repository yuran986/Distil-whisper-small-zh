#!/usr/bin/env bash

python run_eval.py \
  --model_name_or_path "./model" \
  --dataset_name "mozilla-foundation/common_voice_16_1+google/fleurs" \
  --dataset_config_name "zh-CN+cmn_hans_cn" \
  --dataset_split_name "test+test" \
  --text_column_name "sentence+transcription" \
  --batch_size 16 \
  --dtype "bfloat16" \
  --generation_max_length 256 \
  --language "chinese" \
  --attn_implementation "sdpa" \
  --streaming False

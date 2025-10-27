#!/bin/bash

# 에러가 발생해도 다음 줄을 계속 실행합니다.

python /home/hyeonjun/dmllm_codes/run_profile.py \
  --mllm_model_name llavaov \
  --vision_model_name siglip \
  --vision_model_size 6b \
  --llm_model_name qwen2.5 \
  --llm_model_size 72b || true

# python /home/hyeonjun/dmllm_codes/run_profile.py \
#   --mllm_model_name llavaov \
#   --vision_model_name siglip \
#   --vision_model_size 400m \
#   --llm_model_name qwen2.5 \
#   --llm_model_size 7b || true

# python /home/hyeonjun/dmllm_codes/run_profile.py \
#   --mllm_model_name llavaov \
#   --vision_model_name internvit \
#   --vision_model_size 6b \
#   --llm_model_name qwen2.5 \
#   --llm_model_size 72b || true

# python /home/hyeonjun/dmllm_codes/run_profile.py \
#   --mllm_model_name llavaov \
#   --vision_model_name internvit \
#   --vision_model_size 300m \
#   --llm_model_name qwen2.5 \
#   --llm_model_size 72b || true

# python /home/hyeonjun/dmllm_codes/run_profile.py \
#   --mllm_model_name llavaov \
#   --vision_model_name siglip \
#   --vision_model_size 6b \
#   --llm_model_name qwen2.5 \
#   --llm_model_size 32b || true

# python /home/hyeonjun/dmllm_codes/run_profile.py \
#   --mllm_model_name llavaov \
#   --vision_model_name siglip \
#   --vision_model_size 6b \
#   --llm_model_name qwen2.5 \
#   --llm_model_size 7b || true

# python /home/hyeonjun/dmllm_codes/run_profile.py \
#   --mllm_model_name llavaov \
#   --vision_model_name siglip \
#   --vision_model_size 6b \
#   --llm_model_name llama3 \
#   --llm_model_size 70b || true

# python /home/hyeonjun/dmllm_codes/run_profile.py \
#   --mllm_model_name llavaov \
#   --vision_model_name siglip \
#   --vision_model_size 6b \
#   --llm_model_name llama3 \
#   --llm_model_size 8b || true

# python /home/hyeonjun/dmllm_codes/run_profile.py \
#   --mllm_model_name internvl \
#   --vision_model_name internvit \
#   --vision_model_size 6b \
#   --llm_model_name qwen2.5 \
#   --llm_model_size 72b || true

echo "모든 프로파일링 실행이 완료되었습니다."

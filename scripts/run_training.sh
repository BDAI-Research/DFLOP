#!/bin/bash
 
# --- 기본 설정 ---
if [ -z "$1" ]; then
    echo "오류: Node rank를 입력해야 합니다."
    echo "사용법: $0 <rank_number>"
    exit 1
fi
RANKNUM="$1"


torchrun \
  --nnodes=4 \
  --nproc-per-node=8 \
  --master_addr=172.27.53.83 \
  --master_port=25001 \
  --node-rank="$RANKNUM" \
  /giant-data/user/1113870/BDAI/dmllm_codes/ours_llavaov_ver3.py
  # /giant-data/user/1113870/BDAI/dmllm_codes/ours_llavaov_abl.py


#   --dp 2 \
#   --tp 8 \
#   --pp 4 \
#   --llm_model_name=llama3 \
#   --llm_size=70b \
#   --vision_model_name=internvit \
#   --vision_size=6b \
#   --trial 1
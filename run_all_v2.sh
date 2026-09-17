#!/bin/bash
#SBATCH --job-name=all_v2
#SBATCH --partition=gpu_standard
#SBATCH --account=hawshiuan
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_3g.40gb:1
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --output=slurm_v2_%j.log
#SBATCH --error=slurm_v2_%j.err

set -e
cd /groups/hawshiuan/waqar/COMPREHEND_public
export HF_HOME=/groups/hawshiuan/waqar/COMPREHEND_public/.cache_hf
export VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p $HF_HOME

echo "=== Starting vLLM 7B on port 8000 ==="
vllm_env/bin/python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct --port 8000 \
    --max-model-len 8192 --gpu-memory-utilization 0.55 --dtype bfloat16 > vllm_7b_v2.log 2>&1 &
VLLM_7B_PID=$!

for i in {1..90}; do
    if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then echo "7B READY"; break; fi
    sleep 5
done
curl http://localhost:8000/v1/models || (echo "7B FAILED" && cat vllm_7b_v2.log && exit 1)

echo "=== Starting vLLM 3B on port 8001 ==="
vllm_env/bin/python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-3B-Instruct --port 8001 \
    --max-model-len 8192 --gpu-memory-utilization 0.30 --dtype bfloat16 > vllm_3b_v2.log 2>&1 &
VLLM_3B_PID=$!

for i in {1..60}; do
    if curl -s http://localhost:8001/v1/models > /dev/null 2>&1; then echo "3B READY"; break; fi
    sleep 5
done
curl http://localhost:8001/v1/models || (echo "3B FAILED" && cat vllm_3b_v2.log && exit 1)

source comprehen_env/bin/activate

run_one() {
    local name=$1
    echo ""
    echo "#####################################################"
    echo "   RUNNING: $name  at $(date)"
    echo "#####################################################"
    sed -i "s|DATA_FILE = Path(\".*\")|DATA_FILE = Path(\"Results/UF_random_prompts_64_base_${name}.json\")|" main.py
    grep "^DATA_FILE" main.py
    python3 main.py run --method blossom
    python3 evaluation/evaluate.py "Results/UF_random_prompts_64_base_${name}_blossom.json" --model gemma_eval
    echo "   DONE: $name at $(date)"
    echo ""
}

# 4 sequential runs
run_one "similar_v2"
run_one "similar_bestofN_v2"
run_one "dissimilar_v2"
run_one "dissimilar_bestofN_v2"

kill $VLLM_7B_PID $VLLM_3B_PID || true
echo "=== ALL DONE ==="
ls -la Results/*v2*.json
ls -la Results/eval_outputs/*v2*.json

#!/bin/bash
#SBATCH --job-name=512_dis
#SBATCH --partition=gpu_standard
#SBATCH --account=hawshiuan
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_3g.40gb:1
#SBATCH --time=10-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --output=slurm_512_dissimilar_%j.log
#SBATCH --error=slurm_512_dissimilar_%j.err

set -e
cd /groups/hawshiuan/waqar/COMPREHEND_public
export HF_HOME=/groups/hawshiuan/waqar/COMPREHEND_public/.cache_hf
export VLLM_USE_FLASHINFER_SAMPLER=0

echo "=== 512 DISSIMILAR: Starting vLLM 7B (port 8010) ==="
vllm_env/bin/python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct --port 8010 \
    --max-model-len 8192 --gpu-memory-utilization 0.55 --dtype bfloat16 > vllm_7b_512_dis.log 2>&1 &
PID_7B=$!

for i in {1..90}; do
    if curl -s http://localhost:8010/v1/models > /dev/null 2>&1; then echo "7B READY"; break; fi
    sleep 5
done

echo "=== 512 DISSIMILAR: Starting vLLM 3B (port 8011) ==="
vllm_env/bin/python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-3B-Instruct --port 8011 \
    --max-model-len 8192 --gpu-memory-utilization 0.30 --dtype bfloat16 > vllm_3b_512_dis.log 2>&1 &
PID_3B=$!

for i in {1..60}; do
    if curl -s http://localhost:8011/v1/models > /dev/null 2>&1; then echo "3B READY"; break; fi
    sleep 5
done

echo "=== 512 DISSIMILAR: Running tree pipeline ==="
source comprehen_env/bin/activate
python3 main_512_dissimilar.py run --method blossom

echo "=== 512 DISSIMILAR: Evaluation ==="
python3 evaluation/evaluate.py Results/UF_random_prompts_512_base_dissimilar_blossom.json --model gemma_eval

kill $PID_7B $PID_3B || true
echo "=== 512 DISSIMILAR DONE ==="
ls -la Results/UF_random_prompts_512_base_dissimilar_blossom.json

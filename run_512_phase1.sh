#!/bin/bash
#SBATCH --job-name=scale_512_p1
#SBATCH --partition=gpu_standard
#SBATCH --account=hawshiuan
#SBATCH --qos=part_qos_standard
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_3g.40gb:1
#SBATCH --time=10-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --output=slurm_512p1_%j.log
#SBATCH --error=slurm_512p1_%j.err

set -e
cd /groups/hawshiuan/waqar/COMPREHEND_public
export HF_HOME=/groups/hawshiuan/waqar/COMPREHEND_public/.cache_hf
export VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p $HF_HOME

echo "=== Starting vLLM 7B on port 8000 ==="
vllm_env/bin/python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct --port 8000 \
    --max-model-len 8192 --gpu-memory-utilization 0.55 --dtype bfloat16 > vllm_7b_512.log 2>&1 &
VLLM_7B_PID=$!

for i in {1..90}; do
    if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then echo "7B READY"; break; fi
    sleep 5
done
curl http://localhost:8000/v1/models || (echo "7B FAILED" && cat vllm_7b_512.log && exit 1)

echo "=== Starting vLLM 3B on port 8001 ==="
vllm_env/bin/python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-3B-Instruct --port 8001 \
    --max-model-len 8192 --gpu-memory-utilization 0.30 --dtype bfloat16 > vllm_3b_512.log 2>&1 &
VLLM_3B_PID=$!

for i in {1..60}; do
    if curl -s http://localhost:8001/v1/models > /dev/null 2>&1; then echo "3B READY"; break; fi
    sleep 5
done
curl http://localhost:8001/v1/models || (echo "3B FAILED" && cat vllm_3b_512.log && exit 1)

source comprehen_env/bin/activate

echo "=== Running 512-paper Phase 1 tree build ==="
sed -i 's|DATA_FILE = Path(".*")|DATA_FILE = Path("Results/UF_random_prompts_512_base.json")|' main.py
grep "^DATA_FILE" main.py
python3 main.py run --method blossom
python3 evaluation/evaluate.py "Results/UF_random_prompts_512_base_blossom.json" --model gemma_eval

kill $VLLM_7B_PID $VLLM_3B_PID || true
echo "=== 512 JOB DONE ==="
ls -la Results/*512*.json
ls -la Results/eval_outputs/*512*.json

#!/bin/bash
#SBATCH --job-name=sample512
#SBATCH --partition=standard
#SBATCH --account=hawshiuan
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm_sample512_%j.log
#SBATCH --error=slurm_sample512_%j.err

set -e
cd /groups/hawshiuan/waqar/COMPREHEND_public
export HF_HOME=/groups/hawshiuan/waqar/COMPREHEND_public/.cache_hf
mkdir -p $HF_HOME

echo "=== STEP 1: Activate env ==="
source comprehen_env/bin/activate

echo "=== STEP 2: Running sample script ==="
python3 sample_neurips_512.py

echo "=== DONE ==="
ls -la neurips_similar_512.csv neurips_dissimilar_512.csv

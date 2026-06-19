#!/bin/bash
#SBATCH --job-name=CycleSpikeFitter
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=/cluster/scratch/nbrehm/CarbonBoxModel/logs/%A_%a.out
#SBATCH --error=/cluster/scratch/nbrehm/CarbonBoxModel/logs/%A_%a.err
#SBATCH --array=0-100

module load stack/2024-06
module load python/3.11.6
source /cluster/scratch/nbrehm/CarbonBoxModel/venv/bin/activate
cd /cluster/scratch/nbrehm/CarbonBoxModel

START=770
YEAR=$((START + SLURM_ARRAY_TASK_ID))

echo "Running for year: $YEAR"
python MCMC_runner.py --year $YEAR --eventdetrend True
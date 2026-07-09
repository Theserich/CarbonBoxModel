#!/bin/bash
#SBATCH --job-name=CycleSpikeFitter
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=1G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=/cluster/scratch/nbrehm/CarbonBoxModel/logs/%A_%a.out
#SBATCH --error=/cluster/scratch/nbrehm/CarbonBoxModel/logs/%A_%a.err
#SBATCH --array=0-2

module load stack/2024-06
module load python/3.11.6
source /cluster/scratch/nbrehm/CarbonBoxModel/venv/bin/activate
cd /cluster/scratch/nbrehm/CarbonBoxModel

TOTAL_START=-4000
TOTAL_END=2000
TOTAL_YEARS=$((TOTAL_END - TOTAL_START))

CHUNK=10

YEAR_START=$((TOTAL_START + SLURM_ARRAY_TASK_ID * CHUNK))
YEAR_END=$((YEAR_START + CHUNK))

echo "Job $SLURM_ARRAY_TASK_ID: years $YEAR_START to $YEAR_END"
python MCMC_runner.py --year_start $YEAR_START --year_end $YEAR_END --eventdetrend False
#!/bin/bash
#SBATCH --gpus-per-node=v100l:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --account=def-annielee
#SBATCH --array=0-30

module load python/3.11
source ~/env/bin/activate
module load gcc
module load cuda
module load arrow

echo "Job Array ID / Job ID: $SLURM_ARRAY_JOB_ID / $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"

TRANSFER_LANGUAGES=('eng' 'spa' 'deu' 'jpn' 'fra' 'cmn' 'ukr' 'ceb' 'arz' 'ind' 'heb' 'zlm' 'tha' 'dan' 'tgl' 'tam' 'ron' 'ben' 'urd' 'swe' 'hin' 'por' 'ces' 'rus' 'nld' 'pol' 'hrv' 'ita' 'vie' 'eus' 'hun' 'fin' 'srp')

TRANSFER_LANG=${TRANSFER_LANGUAGES[$SLURM_ARRAY_TASK_ID]}
echo "Running transfer language: $TRANSFER_LANG"

python evaluation.py --transfer_lang "$TRANSFER_LANG"

#!/bin/bash
#SBATCH --gpus-per-node=v100l:1
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --account=def-annielee

module load python/3.11
source ~/env/bin/activate

module load gcc
module load cuda
module load arrow

echo "Job Array ID / Job ID: $SLURM_ARRAY_JOB_ID / $SLURM_JOB_ID"

TRANSFER_LANGUAGES=('eng' 'spa' 'deu' 'jpn' 'fra' 'cmn' 'ukr' 'ceb' 'arz' 'ind' 'heb' 'zlm' 'tha' 'dan' 'tgl' 'tam' 'ron' 'ben' 'urd' 'swe' 'hin' 'por' 'ces' 'rus' 'nld' 'pol' 'hrv' 'ita' 'vie' 'eus' 'hun' 'fin' 'srp')
for TRANSFER_LANGUAGE in "${TRANSFER_LANGUAGES[@]}"; do
    echo "Processing language: $TRANSFER_LANGUAGE"
    python trainer.py --lang="$TRANSFER_LANGUAGE"
done
python evaluation.py
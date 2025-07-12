#!/bin/bash
TRANSFER_LANGUAGES=('eng' 'spa' 'deu' 'jpn' 'fra' 'cmn' 'ukr' 'ceb' 'arz' 'ind' 'heb' 'zlm' 'tha' 'dan' 'tgl' 'tam' 'ron' 'ben' 'urd' 'swe' 'hin' 'por' 'ces' 'rus' 'nld' 'pol' 'hrv' 'ita' 'vie' 'eus' 'hun' 'fin' 'srp')
for TRANSFER_LANGUAGE in "${TRANSFER_LANGUAGES[@]}"; do
    echo "Processing language: $TRANSFER_LANGUAGE"
    python trainer.py --lang="$TRANSFER_LANGUAGE"
done

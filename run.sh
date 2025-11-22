#!/bin/bash
#SBATCH --gpus-per-node=h100
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --account=rrg-annielee
#SBATCH --array=0-7
module load python/3.11
source ~/env/bin/activate
module load gcc
module load cuda
module load arrow

echo "Job Array ID / Job ID: $SLURM_ARRAY_JOB_ID / $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"

TRANSFER_LANGUAGES=('eng' 'spa' 'deu' 'jpn' 'fra' 'cmn' 'ukr' 'ceb' 'arz' 'ind' 'heb' 'zlm' 'tha' 'dan' 'tgl' 'tam' 'ron' 'ben' 'urd' 'swe' 'hin' 'por' 'ces' 'rus' 'nld' 'pol' 'hrv' 'ita' 'vie' 'eus' 'hun' 'fin' 'srp')

TOTAL_LANGS=${#TRANSFER_LANGUAGES[@]}
NUM_JOBS=8

# Ceiling division to get chunk size
CHUNK_SIZE=$(( (TOTAL_LANGS + NUM_JOBS - 1) / NUM_JOBS ))

START_INDEX=$(( SLURM_ARRAY_TASK_ID * CHUNK_SIZE ))
END_INDEX=$(( START_INDEX + CHUNK_SIZE ))

echo "Job $SLURM_ARRAY_TASK_ID processing indices $START_INDEX to $END_INDEX"

for (( i=$START_INDEX; i<$END_INDEX && i<$TOTAL_LANGS; i++ ))
do
    LANG=${TRANSFER_LANGUAGES[$i]}
    echo "Running transfer language: $LANG"
    python trainer.py --dataset "taxi1500" --lang "$LANG"
done


#LANGS=('ace_Arab' 'ace_Latn' 'acm_Arab' 'acq_Arab' 'aeb_Arab' 'afr_Latn' 'ajp_Arab' 'aka_Latn' 'als_Latn' 'amh_Ethi' 'apc_Arab' 'arb_Arab' 'arb_Latn' 'ars_Arab' 'ary_Arab' 'arz_Arab' 'asm_Beng' 'ast_Latn' 'awa_Deva' 'ayr_Latn' 'azb_Arab' 'azj_Latn' 'bak_Cyrl' 'bam_Latn' 'ban_Latn' 'bel_Cyrl' 'bem_Latn' 'ben_Beng' 'bho_Deva' 'bjn_Arab' 'bjn_Latn' 'bod_Tibt' 'bos_Latn' 'bug_Latn' 'bul_Cyrl' 'cat_Latn' 'ceb_Latn' 'ces_Latn' 'cjk_Latn' 'ckb_Arab' 'crh_Latn' 'cym_Latn' 'dan_Latn' 'deu_Latn' 'dik_Latn' 'dyu_Latn' 'dzo_Tibt' 'ell_Grek' 'eng_Latn' 'epo_Latn' 'est_Latn' 'eus_Latn' 'ewe_Latn' 'fao_Latn' 'fij_Latn' 'fin_Latn' 'fon_Latn' 'fra_Latn' 'fur_Latn' 'fuv_Latn' 'gaz_Latn' 'gla_Latn' 'gle_Latn' 'glg_Latn' 'grn_Latn' 'guj_Gujr' 'hat_Latn' 'hau_Latn' 'heb_Hebr' 'hin_Deva' 'hne_Deva' 'hrv_Latn' 'hun_Latn' 'hye_Armn' 'ibo_Latn' 'ilo_Latn' 'ind_Latn' 'isl_Latn' 'ita_Latn' 'jav_Latn' 'jpn_Jpan' 'kab_Latn' 'kac_Latn' 'kam_Latn' 'kan_Knda' 'kas_Arab' 'kas_Deva' 'kat_Geor' 'kaz_Cyrl' 'kbp_Latn' 'kea_Latn' 'khk_Cyrl' 'khm_Khmr' 'kik_Latn' 'kin_Latn' 'kir_Cyrl' 'kmb_Latn' 'kmr_Latn' 'knc_Arab' 'knc_Latn' 'kon_Latn' 'kor_Hang' 'lao_Laoo' 'lij_Latn' 'lim_Latn' 'lin_Latn' 'lit_Latn' 'lmo_Latn' 'ltg_Latn' 'ltz_Latn' 'lua_Latn' 'lug_Latn' 'luo_Latn' 'lus_Latn' 'lvs_Latn' 'mag_Deva' 'mai_Deva' 'mal_Mlym' 'mar_Deva' 'min_Arab' 'min_Latn' 'mkd_Cyrl' 'mlt_Latn' 'mni_Beng' 'mos_Latn' 'mri_Latn' 'mya_Mymr' 'nld_Latn' 'nno_Latn' 'nob_Latn' 'npi_Deva' 'nqo_Nkoo' 'nso_Latn' 'nus_Latn' 'nya_Latn' 'oci_Latn' 'ory_Orya' 'pag_Latn' 'pan_Guru' 'pap_Latn' 'pbt_Arab' 'pes_Arab' 'plt_Latn' 'pol_Latn' 'por_Latn' 'prs_Arab' 'quy_Latn' 'ron_Latn' 'run_Latn' 'rus_Cyrl' 'sag_Latn' 'san_Deva' 'sat_Olck' 'scn_Latn' 'shn_Mymr' 'sin_Sinh' 'slk_Latn' 'slv_Latn' 'smo_Latn' 'sna_Latn' 'snd_Arab' 'som_Latn' 'sot_Latn' 'spa_Latn' 'srd_Latn' 'srp_Cyrl' 'ssw_Latn' 'sun_Latn' 'swe_Latn' 'swh_Latn' 'szl_Latn' 'tam_Taml' 'taq_Latn' 'taq_Tfng' 'tat_Cyrl' 'tel_Telu' 'tgk_Cyrl' 'tgl_Latn' 'tha_Thai' 'tir_Ethi' 'tpi_Latn' 'tsn_Latn' 'tso_Latn' 'tuk_Latn' 'tum_Latn' 'tur_Latn' 'twi_Latn' 'tzm_Tfng' 'uig_Arab' 'ukr_Cyrl' 'umb_Latn' 'urd_Arab' 'uzn_Latn' 'vec_Latn' 'vie_Latn' 'war_Latn' 'wol_Latn' 'xho_Latn' 'ydd_Hebr' 'yor_Latn' 'yue_Hant' 'zho_Hans' 'zho_Hant' 'zsm_Latn' 'zul_Latn')
#LANGS=('tso_Latn' 'tuk_Latn' 'tum_Latn' 'tur_Latn' 'twi_Latn' 'tzm_Tfng' 'uig_Arab' 'ukr_Cyrl' 'umb_Latn' 'urd_Arab' 'uzn_Latn' 'vec_Latn' 'vie_Latn' 'war_Latn' 'wol_Latn' 'xho_Latn' 'ydd_Hebr' 'yor_Latn' 'yue_Hant' 'zho_Hans' 'zho_Hant' 'zsm_Latn' 'zul_Latn')
#LANGS1=('ace_Arab' 'ace_Latn' 'acm_Arab' 'acq_Arab' 'aeb_Arab' 'afr_Latn' 'ajp_Arab' 'aka_Latn' 'als_Latn' 'amh_Ethi' 'apc_Arab' 'arb_Arab' 'arb_Latn' 'ars_Arab' 'ary_Arab' 'arz_Arab' 'asm_Beng' 'ast_Latn' 'awa_Deva' 'ayr_Latn' 'azb_Arab' 'azj_Latn' 'bak_Cyrl' 'bam_Latn' 'ban_Latn' 'bel_Cyrl' 'bem_Latn' 'ben_Beng' 'bho_Deva' 'bjn_Arab' 'bjn_Latn' 'bod_Tibt' 'bos_Latn' 'bug_Latn' 'bul_Cyrl' 'cat_Latn' 'ceb_Latn' 'ces_Latn' 'cjk_Latn' 'ckb_Arab' 'crh_Latn' 'cym_Latn' 'dan_Latn' 'deu_Latn' 'dik_Latn' 'dyu_Latn' 'dzo_Tibt' 'ell_Grek' 'eng_Latn' 'epo_Latn' 'est_Latn' 'eus_Latn' 'ewe_Latn' 'fao_Latn' 'fij_Latn' 'fin_Latn' 'fon_Latn' 'fra_Latn' 'fur_Latn' 'fuv_Latn' 'gaz_Latn' 'gla_Latn' 'gle_Latn' 'glg_Latn' 'grn_Latn' 'guj_Gujr' 'hat_Latn' 'hau_Latn')
#LANGS2=('heb_Hebr' 'hin_Deva' 'hne_Deva' 'hrv_Latn' 'hun_Latn' 'hye_Armn' 'ibo_Latn' 'ilo_Latn' 'ind_Latn' 'isl_Latn' 'ita_Latn' 'jav_Latn' 'jpn_Jpan' 'kab_Latn' 'kac_Latn' 'kam_Latn' 'kan_Knda' 'kas_Arab' 'kas_Deva' 'kat_Geor' 'kaz_Cyrl' 'kbp_Latn' 'kea_Latn' 'khk_Cyrl' 'khm_Khmr' 'kik_Latn' 'kin_Latn' 'kir_Cyrl' 'kmb_Latn' 'kmr_Latn' 'knc_Arab' 'knc_Latn' 'kon_Latn' 'kor_Hang' 'lao_Laoo' 'lij_Latn' 'lim_Latn' 'lin_Latn' 'lit_Latn' 'lmo_Latn' 'ltg_Latn' 'ltz_Latn' 'lua_Latn' 'lug_Latn' 'luo_Latn' 'lus_Latn' 'lvs_Latn' 'mag_Deva' 'mai_Deva' 'mal_Mlym' 'mar_Deva' 'min_Arab' 'min_Latn' 'mkd_Cyrl' 'mlt_Latn' 'mni_Beng' 'mos_Latn' 'mri_Latn' 'mya_Mymr' 'nld_Latn' 'nno_Latn' 'nob_Latn' 'npi_Deva' 'nqo_Nkoo' 'nso_Latn' 'nus_Latn' 'nya_Latn' 'oci_Latn')
#LANGS3=('ory_Orya' 'pag_Latn' 'pan_Guru' 'pap_Latn' 'pbt_Arab' 'pes_Arab' 'plt_Latn' 'pol_Latn' 'por_Latn' 'prs_Arab' 'quy_Latn' 'ron_Latn' 'run_Latn' 'rus_Cyrl' 'sag_Latn' 'san_Deva' 'sat_Olck' 'scn_Latn' 'shn_Mymr' 'sin_Sinh' 'slk_Latn' 'slv_Latn' 'smo_Latn' 'sna_Latn' 'snd_Arab' 'som_Latn' 'sot_Latn' 'spa_Latn' 'srd_Latn' 'srp_Cyrl' 'ssw_Latn' 'sun_Latn' 'swe_Latn' 'swh_Latn' 'szl_Latn' 'tam_Taml' 'taq_Latn' 'taq_Tfng' 'tat_Cyrl' 'tel_Telu' 'tgk_Cyrl' 'tgl_Latn' 'tha_Thai' 'tir_Ethi' 'tpi_Latn' 'tsn_Latn' 'tso_Latn' 'tuk_Latn' 'tum_Latn' 'tur_Latn' 'twi_Latn' 'tzm_Tfng' 'uig_Arab' 'ukr_Cyrl' 'umb_Latn' 'urd_Arab' 'uzn_Latn' 'vec_Latn' 'vie_Latn' 'war_Latn' 'wol_Latn' 'xho_Latn' 'ydd_Hebr' 'yor_Latn' 'yue_Hant' 'zho_Hans' 'zho_Hant' 'zsm_Latn' 'zul_Latn')
#LANGS=('zsm_Latn' 'zul_Latn')
#if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
#  for lang in "${LANGS[@]}"; do
#    python evaluation.py --dataset sib200  --transfer_lang=$lang
##     python trainer.py --dataset sib200 --lang=$lang
#  done
#elif [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
#  for lang in "${LANGS2[@]}"; do
#    python evaluation.py --dataset sib200 --transfer_lang=$lang
##     python trainer.py --dataset sib200 --lang=$lang
#  done
#else
#  for lang in "${LANGS3[@]}"; do
#    python evaluation.py --dataset sib200 --transfer_lang=$lang
##     python trainer.py --dataset sib200 --lang=$lang
#  done
#fi


import trainer

TRANSFER_LANGUAGES = ['eng', 'spa', 'deu', 'jpn', 'fra', 'cmn', 'ukr', 'ceb', 'arz', 'ind', 'heb', 'zlm', 'tha', 'dan', 'tgl', 'tam', 'ron', 'ben', 'urd', 'swe', 'hin', 'por', 'ces', 'rus', 'nld', 'pol', 'hrv', 'ita', 'vie', 'eus', 'hun', 'fin', 'srp']


if __name__ == "__main__":
    print('Starting training for all languages...')
    for language in TRANSFER_LANGUAGES:
        print(f"Training for language: {language}")
        try:
            trainer.finetune(language)
            print(f"Training completed for language: {language}")
        except Exception as e:
            print(f"Error during training for language {language}: {e}")
    print('All training processes completed.')

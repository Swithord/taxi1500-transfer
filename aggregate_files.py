import pandas as pd
import os

# TRANSFER_LANGUAGES = ['eng', 'spa', 'deu', 'jpn', 'fra', 'cmn', 'ukr', 'ceb', 'arz', 'ind', 'heb', 'zlm', 'tha', 'dan', 'tgl', 'tam', 'ron', 'ben', 'urd', 'swe', 'hin', 'por', 'ces', 'rus', 'nld', 'pol', 'hrv', 'ita', 'vie', 'eus', 'hun', 'fin', 'srp']


def aggregate_results():
    """
    Aggregate results from multiple evaluation files into a single DataFrame.
    """
    results = []

    for file_path in os.listdir('results'):
        # file_path = f'evaluation_results_{lang}.csv'
        #if not os.path.exists(file_path):
        #    print(f"File {file_path} does not exist. Skipping.")
        #    continue
        lang = file_path.split('_')[-2]
        df = pd.read_csv('results/' + file_path)
        df['transfer_language'] = lang
        results.append(df)

    aggregated_df = pd.concat(results, ignore_index=True)
    aggregated_df.to_csv('evaluation_results.csv', index=False)


if __name__ == "__main__":
    aggregate_results()

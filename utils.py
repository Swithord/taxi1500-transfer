from datasets import Dataset, DatasetDict
import pandas as pd


def preprocess_data(language: str, labels: dict[str, int], splits: set[str]) -> DatasetDict:
    dataset = DatasetDict()
    for split in splits:
        df = pd.read_csv(f"data/{language}_{split}.csv", index_col=0)
        df['label'] = df['classification'].map(labels)
        df = df.drop(columns=['classification'])
        dataset[split] = Dataset.from_pandas(df)

    return dataset

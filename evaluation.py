from datasets import DatasetDict
from transformers import BertForSequenceClassification, BertTokenizer, Trainer
from utils import preprocess_data
import evaluate
import os
import pandas as pd
import warnings
from tqdm import tqdm

TRANSFER_LANGUAGES = ['eng', 'spa', 'deu', 'jpn', 'fra', 'cmn', 'ukr', 'ceb', 'arz', 'ind', 'heb', 'zlm', 'tha', 'dan', 'tgl', 'tam', 'ron', 'ben', 'urd', 'swe', 'hin', 'por', 'ces', 'rus', 'nld', 'pol', 'hrv', 'ita', 'vie', 'eus', 'hun', 'fin', 'srp']


def evaluate_model(dataset: DatasetDict, model: BertForSequenceClassification, tokenizer: BertTokenizer, task_lang: str, transfer_lang: str):
    """
    Evaluate the model on a specific task language, optionally transferring from another language.

    Args:
        dataset (DatasetDict): Preprocessed dataset containing 'train', 'dev', and 'test' splits.
        model (BertForSequenceClassification): Pretrained model for sequence classification.
        tokenizer (BertTokenizer): Tokenizer for the model.
        task_lang (str): Language of the task to evaluate.
        transfer_lang (str): Language from which to transfer knowledge.

    Returns:
        dict: Evaluation results including accuracy and F1 score.
    """
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
    )
    predictions = trainer.predict(tokenized_datasets['test'])
    preds = predictions.predictions.argmax(axis=-1)
    labels = tokenized_datasets['test']['label']

    accuracy = evaluate.load('accuracy')
    f1 = evaluate.load('f1')
    acc = accuracy.compute(predictions=preds, references=labels)
    f1_score = f1.compute(predictions=preds, references=labels, average='weighted')
    print(task_lang, transfer_lang, acc, f1_score)

    return {
        'accuracy': acc['accuracy'],
        'f1_score': f1_score['f1']
    }


def main():
    warnings.filterwarnings("ignore", message=".*pin_memory.*")
    languages = set()
    for file in os.listdir('data'):
        languages.add(file.split('_')[0])

    results = {
        'task_lang': [],
        'transfer_lang': [],
        'accuracy': [],
        'f1_score': []
    }

    labels = {'Recommendation': 0, 'Faith': 1, 'Description': 2, 'Sin': 3, 'Grace': 4, 'Violence': 5}
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

    models = {}
    for transfer_lang in TRANSFER_LANGUAGES:
        model_path = f'models/{transfer_lang}'
        if not os.path.exists(model_path):
            print(f"Model for {transfer_lang} not found, skipping.")
            continue
        models[transfer_lang] = BertForSequenceClassification.from_pretrained(model_path, num_labels=len(labels))

    test_datasets = {}
    for task_lang in languages:
        test_datasets[task_lang] = preprocess_data(task_lang, labels, {'test'})

    for task_lang in tqdm(languages):
        for transfer_lang in TRANSFER_LANGUAGES:
            if task_lang not in test_datasets or transfer_lang not in models:
                print(f"Skipping evaluation for {task_lang} with transfer from {transfer_lang}.")
                continue

            transfer_results = evaluate_model(
                dataset=test_datasets[task_lang],
                model=models[transfer_lang],
                tokenizer=tokenizer,
                task_lang=task_lang,
                transfer_lang=transfer_lang
            )
            results['task_lang'].append(task_lang)
            results['transfer_lang'].append(transfer_lang)
            results['accuracy'].append(transfer_results['accuracy'])
            results['f1_score'].append(transfer_results['f1_score'])

    df_results = pd.DataFrame(results)
    df_results.to_csv('evaluation_results.csv', index=False)


if __name__ == "__main__":
    main()

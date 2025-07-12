from transformers import BertForSequenceClassification, BertTokenizer, Trainer
from utils import preprocess_data
import evaluate
import os
import pandas as pd
import warnings
from tqdm import tqdm

TRANSFER_LANGUAGES = ['eng', 'spa', 'deu', 'jpn', 'fra', 'cmn']


def evaluate_model(task_lang: str, transfer_lang: str = '') -> dict:
    """
    Evaluate the model on a specific task language, optionally transferring from another language.

    Args:
        task_lang (str): The language code for the task language.
        transfer_lang (str): The language code for the transfer language (default is empty).

    Returns:
        dict: Evaluation results including accuracy and F1 score.
    """
    labels = {'Recommendation': 0, 'Faith': 1, 'Description': 2, 'Sin': 3, 'Grace': 4, 'Violence': 5}
    dataset = preprocess_data(task_lang, labels, {'test'})

    if transfer_lang:
        model = BertForSequenceClassification.from_pretrained(f'models/{transfer_lang}', num_labels=len(labels))
        tokenizer = BertTokenizer.from_pretrained(f'models/{transfer_lang}')
    else:
        model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=len(labels))
        tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

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

    for task_lang in tqdm(languages):
        for transfer_lang in TRANSFER_LANGUAGES:
            transfer_results = evaluate_model(task_lang, transfer_lang)
            results['task_lang'].append(task_lang)
            results['transfer_lang'].append(transfer_lang)
            results['accuracy'].append(transfer_results['accuracy'])
            results['f1_score'].append(transfer_results['f1_score'])

    df_results = pd.DataFrame(results)
    df_results.to_csv('evaluation_results.csv', index=False)


if __name__ == "__main__":
    main()

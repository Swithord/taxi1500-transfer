from datasets import DatasetDict
from transformers import BertForSequenceClassification, BertTokenizer, Trainer
from utils import preprocess_data
import evaluate
import os
import pandas as pd
import warnings
from tqdm import tqdm
from argparse import ArgumentParser

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
    parser = ArgumentParser(description="Evaluate multilingual BERT models on various languages.")
    parser.add_argument('--transfer_language', type=str, choices=TRANSFER_LANGUAGES, default='eng',
                        help="Language to transfer knowledge from (default: 'eng').")
    args = parser.parse_args()

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
    model = BertForSequenceClassification.from_pretrained(f'models/{args.transfer_language}', num_labels=len(labels))

    test_datasets = {}
    for task_lang in languages:
        test_datasets[task_lang] = preprocess_data(task_lang, labels, {'test'})

    for task_lang in tqdm(languages):
        transfer_results = evaluate_model(
            dataset=test_datasets[task_lang],
            model=model,
            tokenizer=tokenizer,
            task_lang=task_lang,
            transfer_lang=args.transfer_language
        )
        results['task_lang'].append(task_lang)
        results['transfer_lang'].append(args.transfer_language)
        results['accuracy'].append(transfer_results['accuracy'])
        results['f1_score'].append(transfer_results['f1_score'])

    df_results = pd.DataFrame(results)
    df_results.to_csv(f'evaluation_results_{args.transfer_language}.csv', index=False)


if __name__ == "__main__":
    main()
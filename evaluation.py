from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import BertForSequenceClassification, BertTokenizer, AutoModelForSequenceClassification, Trainer, AutoTokenizer
from utils import preprocess_data
import evaluate
import os
import pandas as pd
import warnings
from tqdm import tqdm
from argparse import ArgumentParser

TRANSFER_LANGUAGES = ['eng', 'spa', 'deu', 'jpn', 'fra', 'cmn', 'ukr', 'ceb', 'arz', 'ind', 'heb', 'zlm', 'tha', 'dan', 'tgl', 'tam', 'ron', 'ben', 'urd', 'swe', 'hin', 'por', 'ces', 'rus', 'nld', 'pol', 'hrv', 'ita', 'vie', 'eus', 'hun', 'fin', 'srp']


def evaluate_model(dataset: DatasetDict, model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer, task_lang: str, transfer_lang: str):
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
    f1_score = f1.compute(predictions=preds, references=labels, average='macro')
    print(task_lang, transfer_lang, acc, f1_score)

    return {
        'accuracy': acc['accuracy'],
        'f1_score': f1_score['f1']
    }


def main():
    parser = ArgumentParser(description="Evaluate multilingual BERT models on various languages.")
    parser.add_argument('--transfer_language', type=str, default='eng',
                        help="Language to transfer knowledge from (default: 'eng').")
    parser.add_argument('--dataset', type=str, choices=['taxi1500', 'sib200'], default='taxi1500',)
    args = parser.parse_args()

    results = {
        'task_lang': [],
        'transfer_lang': [],
        'accuracy': [],
        'f1_score': []
    }

    languages = set()

    if args.dataset == 'taxi1500':
        for file in os.listdir('data'):
            languages.add(file.split('_')[0])

        labels = {'Recommendation': 0, 'Faith': 1, 'Description': 2, 'Sin': 3, 'Grace': 4, 'Violence': 5}
        tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        model = AutoModelForSequenceClassification.from_pretrained(f'models/{args.transfer_language}', num_labels=len(labels))

        test_datasets = {}
        for task_lang in languages:
            test_datasets[task_lang] = preprocess_data(task_lang, labels, {'test'})
    elif args.dataset == 'sib200':
        for file in os.listdir('models'):
            languages.add(file)

        labels = ['science/technology', 'travel', 'politics', 'sports', 'health', 'entertainment', 'geography']
        label2id = {label: idx for idx, label in enumerate(labels)}
        model = AutoModelForSequenceClassification.from_pretrained(f'models/{args.transfer_language}', num_labels=len(labels))
        tokenizer = AutoTokenizer.from_pretrained(f'models/{args.transfer_language}')

        test_datasets = {}
        for task_lang in languages:
            test_datasets[task_lang] = load_from_disk(f'sib200/{task_lang}')['test']

            def encode_labels(example):
                example["label"] = label2id[example["category"]]
                return example

            test_datasets[task_lang] = DatasetDict({'test': test_datasets[task_lang].map(encode_labels, num_proc=4)})
    else:
        raise ValueError("Unsupported dataset. Choose either 'taxi1500' or 'sib200'.")

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
    df_results.to_csv(f'results/evaluation_results_{args.transfer_language}.csv', index=False)


if __name__ == "__main__":
    main()

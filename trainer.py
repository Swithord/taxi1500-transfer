from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from datasets import Dataset, DatasetDict
import evaluate
import pandas as pd
from argparse import ArgumentParser

TRANSFER_LANGUAGES = ['spa', 'deu', 'jpn', 'fra', 'cmn', 'ukr', 'ceb', 'arz', 'ind', 'heb', 'zlm', 'tha', 'dan', 'tgl', 'tam', 'ron', 'ben', 'urd', 'swe', 'hin', 'por', 'ces', 'rus', 'nld', 'pol', 'hrv', 'ita', 'vie', 'eus', 'hun', 'fin', 'srp']


def preprocess_data(language: str, labels: dict[str, int]) -> DatasetDict:
    train_df = pd.read_csv(f"data/{language}_train.csv", index_col=0)
    val_df = pd.read_csv(f"data/{language}_dev.csv", index_col=0)

    train_df['label'] = train_df['classification'].map(labels)
    val_df['label'] = val_df['classification'].map(labels)

    train_df = train_df.drop(columns=['classification'])
    val_df = val_df.drop(columns=['classification'])

    dataset = DatasetDict({
        'train': Dataset.from_pandas(train_df),
        'validation': Dataset.from_pandas(val_df)
    })

    return dataset


def finetune(language: str):
    labels = {'Recommendation': 0, 'Faith': 1, 'Description': 2, 'Sin': 3, 'Grace': 4, 'Violence': 5}
    dataset = preprocess_data(language, labels)

    model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased",
                                                          num_labels=len(labels))
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.argmax(axis=-1)
        acc = accuracy.compute(predictions=predictions, references=labels)
        f1_score = f1.compute(predictions=predictions, references=labels, average='weighted')
        return {**acc, **f1_score}

    training_args = TrainingArguments(
        output_dir=f"models/{language}",
        learning_rate=1e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=10,
        weight_decay=0.01,
        warmup_steps=50,
        logging_dir=f"logs/{language}",
        eval_strategy="steps",
        eval_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    trainer.train()
    trainer.save_model(f"models/{language}")
    tokenizer.save_pretrained(f"models/{language}")


def main():
    parser = ArgumentParser(description="Fine-tune mBERT model on a specific language in Taxi1500.")
    parser.add_argument('--lang', type=str, required=True, choices=TRANSFER_LANGUAGES,
                        help='Language to fine-tune the model on.')
    args = parser.parse_args()

    finetune(args.language)

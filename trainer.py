from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
import evaluate
from argparse import ArgumentParser
from utils import preprocess_data

TRANSFER_LANGUAGES = ['eng', 'spa', 'deu', 'jpn', 'fra', 'cmn', 'ukr', 'ceb', 'arz', 'ind', 'heb', 'zlm', 'tha', 'dan', 'tgl', 'tam', 'ron', 'ben', 'urd', 'swe', 'hin', 'por', 'ces', 'rus', 'nld', 'pol', 'hrv', 'ita', 'vie', 'eus', 'hun', 'fin', 'srp']


def finetune(language: str):
    labels = {'Recommendation': 0, 'Faith': 1, 'Description': 2, 'Sin': 3, 'Grace': 4, 'Violence': 5}
    dataset = preprocess_data(language, labels, {'train', 'dev'})

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
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=20,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_dir=f"logs/{language}",
        eval_strategy="steps",
        eval_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['dev'],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    trainer.train()
    trainer.save_model(f"models/{language}")
    tokenizer.save_pretrained(f"models/{language}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Fine-tune mBERT model on a specific language in Taxi1500.")
    parser.add_argument('--lang', type=str, required=True, choices=TRANSFER_LANGUAGES,
                        help='Language to fine-tune the model on.')
    args = parser.parse_args()

    finetune(args.lang)

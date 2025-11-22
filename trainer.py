from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
import evaluate
from argparse import ArgumentParser
from datasets import load_dataset
from utils import preprocess_data
from peft import LoraConfig, get_peft_model

TRANSFER_LANGUAGES = ['eng', 'spa', 'deu', 'jpn', 'fra', 'cmn', 'ukr', 'ceb', 'arz', 'ind', 'heb', 'zlm', 'tha', 'dan', 'tgl', 'tam', 'ron', 'ben', 'urd', 'swe', 'hin', 'por', 'ces', 'rus', 'nld', 'pol', 'hrv', 'ita', 'vie', 'eus', 'hun', 'fin', 'srp']

def finetune_taxi1500(language: str):
    labels = {'Recommendation': 0, 'Faith': 1, 'Description': 2, 'Sin': 3, 'Grace': 4, 'Violence': 5}
    labels_map = {label: idx for idx, label in enumerate(labels)}
    dataset = preprocess_data(language, labels_map, {'train', 'dev'})

    # model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased",
    #                                                       num_labels=len(labels))
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        device_map="auto",  # automatically place on GPU
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        task_type="SEQ_CLS"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

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


def finetune_sib200(language: str):
    labels = ['science/technology', 'travel', 'politics', 'sports', 'health', 'entertainment', 'geography']
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    dataset = load_dataset('Davlan/sib200', language)

    model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=len(labels))
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    def encode_labels(example):
        example["label"] = label2id[example["category"]]
        return example

    dataset = dataset.map(encode_labels)

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=256)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.argmax(axis=-1)
        acc = accuracy.compute(predictions=predictions, references=labels)
        f1_score = f1.compute(predictions=predictions, references=labels, average='macro')
        return {**acc, **f1_score}

    training_args = TrainingArguments(
        output_dir=f"models/{language}",
        learning_rate=2e-5,
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
        eval_dataset=tokenized_datasets['validation'],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    trainer.train()
    trainer.save_model(f"models/{language}")
    tokenizer.save_pretrained(f"models/{language}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Fine-tune model on a specific language in Taxi1500.")
    parser.add_argument('--lang', type=str, required=True,
                        help='Language to fine-tune the model on.')
    parser.add_argument('--dataset', type=str, required=True, choices=('taxi1500', 'sib200'))
    args = parser.parse_args()

    if args.dataset == 'taxi1500':
        finetune_taxi1500(args.lang)
    elif args.dataset == 'sib200':
        finetune_sib200(args.lang)
    else:
        print(f"Dataset {args.dataset} is not supported. Please choose from: taxi1500, sib200.")

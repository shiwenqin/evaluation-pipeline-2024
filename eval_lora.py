# This script was adapted from `LoRA.ipynb` in the HuggingFace PEFT repository:
# https://github.com/huggingface/peft/blob/main/examples/sequence_classification/LoRA.ipynb
import argparse
import os
import numpy as np
from copy import deepcopy

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

import evaluate
import hydra
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer,
    get_linear_schedule_with_warmup, set_seed
)
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    LoraConfig,
    PeftType
)
from tqdm import tqdm
from transformers import Trainer, TrainingArguments

def load_tokenizer(tokenizer_path, padding_side):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side=padding_side,
                                              trust_remote_code=True)
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer

def tokenize_fn(examples, tokenizer, input_columns=["sentence"], max_length=128):
    if len(input_columns) == 1:
        return tokenizer(examples[input_columns[0]], truncation=True, max_length=max_length)
    elif len(input_columns) == 2:
        return tokenizer(examples[input_columns[0]], examples[input_columns[1]],
                         truncation=True, max_length=max_length)
    else:
        raise ValueError(f"Bad number of input_columns: {len(input_columns)}")
    
def collate_fn(examples):
    return tokenizer.pad(examples, padding="longest", return_tensors="pt")

def train_task(task_name, task_cfg, tokenizer, peft_config, cfg):
    
    data_files = {"train": f"evaluation_data/glue_filtered/{task_name}.train.jsonl",
                  "validation": f"evaluation_data/glue_filtered/{task_name}.valid.jsonl"}
    dataset = load_dataset("json", data_files=data_files)

    if task_name == "multirc":
        dataset = dataset.map(lambda example: {'question_and_answer': f"{example['question']} {example['answer']}"},
                                     remove_columns=['question', 'answer'])
    elif task_name == "wsc":
        dataset = dataset.map(lambda example: {'span1_and_span2_text':
                                               f"Does \"{example['span2_text']}\" refer to \"{example['span1_text']}\"?"},
                                     remove_columns=['span1_text', 'span2_text'])

    metric = evaluate.load(task_cfg.collection, task_name)

    if task_name == "multirc":
        metric = evaluate.load(task_cfg.collection, "wsc")  # don't use `f1_m` or `f1_a`; just use `accuracy`, as in "wsc"

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    
    input_columns = task_cfg.structure
    columns_to_remove = task_cfg.structure + ["idx"]

    tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=columns_to_remove,
                                    fn_kwargs={"tokenizer": tokenizer,
                                               "input_columns": input_columns,
                                               "max_length": task_cfg.max_length})
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    num_labels = len(np.unique(tokenized_dataset["train"]["labels"]))

    if task_name == "mnli":
        dataset_mm = load_dataset("json", data_files={"validation": "evaluation_data/glue_filtered/mnli-mm.valid.jsonl"})
        tokenized_dataset_mm = dataset_mm.map(tokenize_fn, batched=True, remove_columns=columns_to_remove,
                                              fn_kwargs={"tokenizer": tokenizer,
                                                        "input_columns": input_columns,
                                                        "max_length": task_cfg.max_length})
        tokenized_dataset_mm = tokenized_dataset_mm.rename_column("label", "labels")
    
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_path, return_dict=True,
                                                               num_labels=num_labels, trust_remote_code=True)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to('cuda')
    lora_model = get_peft_model(model, peft_config)

    if task_name == "mnli":
        eval_dataset = {"mnli-matched": tokenized_dataset["validation"],
                        "mnli-mismatched": tokenized_dataset_mm["validation"]}
        metric_to_track = "mnli-matched_loss"
    else:
        eval_dataset = tokenized_dataset["validation"]
        metric_to_track = "loss"

    trainer = Trainer(
        model=lora_model,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        args=TrainingArguments(
            output_dir=cfg.output_dir,
            num_train_epochs=task_cfg.num_epochs,
            per_device_train_batch_size=task_cfg.batch_size,
            per_device_eval_batch_size=task_cfg.batch_size,
            evaluation_strategy="epoch",
            save_strategy="no",      # set to "no" if you don't want checkpoints
            logging_strategy="epoch",
            learning_rate=task_cfg.lr,
            optim="adamw_torch",
            metric_for_best_model=metric_to_track,
            warmup_steps=(task_cfg.warmup_proportion * len(dataset["train"]) * task_cfg.num_epochs),
            #load_best_model_at_end=True,
        )
    )

    trainer.train()
    metrics = trainer.evaluate(eval_dataset)
    return metrics

@hydra.main(version_base=None, config_path="configs", config_name="glue")
def main(cfg):

    peft_config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1, modules_to_save=["classifier"])

    # load tokenizer and preprocess dataset 
    tokenizer = load_tokenizer(cfg.model_path, cfg.padding_side)

    for task_name, task_cfg in cfg.tasks.items():
        metrics = train_task(task_name, task_cfg, tokenizer, peft_config, cfg)
        print(f"Task: {task_name}, Metrics: {metrics}")


if __name__ == "__main__":
    main()
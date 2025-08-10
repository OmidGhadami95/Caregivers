# -*- coding: utf-8 -*-
"""ClinicalBERT_LOSO_CV.py"""

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
import torch
from datasets import Dataset
from tqdm import tqdm

# Load preprocessed data
survey_df = pd.read_csv("preprocessed_narrative.csv")

# LOSO-CV Setup
subject_ids = survey_df['ID'].unique()
all_predictions = []
all_true_labels = []

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Main LOSO-CV loop
for test_subject in tqdm(subject_ids, desc="Processing LOSO-CV folds"):
    # Split data
    train_df = survey_df[survey_df['ID'] != test_subject].copy()
    test_df = survey_df[survey_df['ID'] == test_subject].copy()

    # Create datasets using original text
    train_dataset = Dataset.from_pandas(train_df[['UNI', 'High Burden']]).rename_column('High Burden', 'labels')
    test_dataset = Dataset.from_pandas(test_df[['UNI', 'High Burden']]).rename_column('High Burden', 'labels')

    # Initialize model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model = AutoModelForSequenceClassification.from_pretrained(
        "emilyalsentzer/Bio_ClinicalBERT",
        num_labels=2
    ).to(device)

    # Tokenization function
    def tokenize_function(batch):
        return tokenizer(
            batch["UNI"],  # Use original text column
            padding="max_length",
            truncation=True,
            max_length=256
        )

    # Tokenize datasets
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    # Set format
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"./results_{test_subject}",
        num_train_epochs=10,
        per_device_train_batch_size=4,
        learning_rate=2e-5,
        weight_decay=0.0,
        lr_scheduler_type="constant",
        logging_dir=f"./logs_{test_subject}",
        logging_steps=10,
        save_strategy="no",
        fp16=torch.cuda.is_available(),
        warmup_ratio=0.0,
        warmup_steps=0,
    )

    # Custom metrics
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        return {"f1": f1_score(labels, preds, average="binary")}

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        compute_metrics=compute_metrics
    )

    # Train and predict
    trainer.train()
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)

    # Store results
    all_predictions.extend(preds)
    all_true_labels.extend(test_df['High Burden'].tolist())

# Final evaluation
final_f1 = f1_score(all_true_labels, all_predictions, average="binary")
print(f"\nFinal LOSO-CV F1 Score: {final_f1:.4f}")
final_precision = precision_score(all_true_labels, all_predictions, average="binary", zero_division=1)
final_recall = recall_score(all_true_labels, all_predictions, average="binary", zero_division=1)

print(f"\nFinal LOSO-CV F1 Score: {final_f1:.4f}")
print(f"Final LOSO-CV Precision: {final_precision:.4f}")
print(f"Final LOSO-CV Recall: {final_recall:.4f}")
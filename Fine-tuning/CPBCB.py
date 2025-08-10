# -*- coding: utf-8 -*-
"""BERT with Tabular Features Only - LOSO-CV"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import f1_score
from transformers import (
    BertConfig,
    Trainer,
    TrainingArguments,
    PreTrainedModel
)
from torch.utils.data import Dataset as TorchDataset
from transformers.modeling_outputs import SequenceClassifierOutput

# =======================
# Custom Dataset
# =======================
class TabularOnlyDataset(TorchDataset):
    def __init__(self, tabular_data, labels):
        self.tabular = torch.tensor(tabular_data.astype(np.float32).values)
        self.labels = torch.tensor(labels.values, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'tabular_feats': self.tabular[idx],
            'labels': self.labels[idx]
        }

# =======================
# Custom Model
# =======================
class TabularClassifier(PreTrainedModel):
    def __init__(self, config, tabular_dim, tabular_hidden=64):
        super().__init__(config)
        self.classifier = nn.Linear(tabular_dim, config.num_labels)
        self.config = config

    def forward(self, tabular_feats, labels=None):
        logits = self.classifier(tabular_feats)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)


# =======================
# Load and Prepare Data
# =======================
zbi_df = pd.read_csv("PHQ_preprocessed.csv")

# =======================
# Define Parameters
# =======================
target_column = "PHQ_score"
tabular_columns = [
    'Age', 'Sex', 'Race', 'Ethnicity', 'Location', 'Employment', 'Education',
    'Income', 'Health Insurance', 'Relationship to Loved One', 'Years Caring',
    'Daily Caregiving Hours'
]

# =======================
# Load Config
# =======================
config = BertConfig(num_labels=2)

# =======================
# LOSO-CV Training
# =======================
subject_ids = zbi_df['Participant ID'].unique()
all_preds, all_labels = [], []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for test_id in tqdm(subject_ids, desc="LOSO-CV"):
    train_df = zbi_df[zbi_df['Participant ID'] != test_id]
    test_df = zbi_df[zbi_df['Participant ID'] == test_id]

    train_dataset = TabularOnlyDataset(train_df[tabular_columns], train_df[target_column])
    test_dataset = TabularOnlyDataset(test_df[tabular_columns], test_df[target_column])

    model = TabularClassifier(config, tabular_dim=len(tabular_columns))
    model.to(device)

    training_args = TrainingArguments(
        output_dir=f"./results_{test_id}",
        num_train_epochs=10,
        per_device_train_batch_size=4,
        learning_rate=2e-5,
        weight_decay=0.0,
        logging_dir=f"./logs_{test_id}",
        logging_steps=10,
        save_strategy="no",
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    def compute_metrics(eval_pred):
        preds = np.argmax(eval_pred.predictions, axis=1)
        return {"f1": f1_score(eval_pred.label_ids, preds, average="binary")}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)

    all_preds.extend(preds)
    all_labels.extend(test_df[target_column].tolist())

final_f1 = f1_score(all_labels, all_preds, average="binary")
print(f"\nFinal LOSO-CV F1 Score: {final_f1:.4f}")
final_precision = precision_score(all_labels, all_preds, average="binary", zero_division=1)
final_recall = recall_score(all_labels, all_preds, average="binary", zero_division=1)

print(f"\nFinal LOSO-CV F1 Score: {final_f1:.4f}")
print(f"Final LOSO-CV Precision: {final_precision:.4f}")
print(f"Final LOSO-CV Recall: {final_recall:.4f}")

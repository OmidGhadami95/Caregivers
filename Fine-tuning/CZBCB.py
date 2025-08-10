# -*- coding: utf-8 -*-
"""BERT with Tabular Features - LOSO-CV without Feature Selection"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import f1_score
from transformers import (
    BertTokenizer,
    BertConfig,
    Trainer,
    TrainingArguments,
    PreTrainedModel,
    BertModel
)
from torch.utils.data import Dataset as TorchDataset
from transformers.modeling_outputs import SequenceClassifierOutput

# =======================
# Custom Dataset
# =======================
class TextTabularDataset(TorchDataset):
    def __init__(self, text_encodings, tabular_data, labels):
        self.encodings = text_encodings
        self.tabular = torch.tensor(tabular_data.values, dtype=torch.float32)
        self.labels = torch.tensor(labels.values, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['tabular_feats'] = self.tabular[idx]
        item['labels'] = self.labels[idx]
        return item

# =======================
# Custom Model
# =======================
class BertWithTabular(PreTrainedModel):
    def __init__(self, config, tabular_dim):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size + tabular_dim, config.num_labels)
        self.config = config

    def forward(self, input_ids, attention_mask, tabular_feats, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(outputs.pooler_output)
        combined = self.dropout(torch.cat((pooled_output, tabular_feats), dim=1))
        logits = self.classifier(combined)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)

# =======================
# Load and Merge Data
# =======================
narrative_df = pd.read_csv("preprocessed_narrative.csv")
zbi_df = pd.read_csv("ZBI_preprocessed.csv")
data_df = pd.merge(narrative_df, zbi_df, left_on="ID", right_on="Participant ID")

# =======================
# Define Parameters
# =======================
text_column = "UNI"
target_column = "High Burden"
tabular_columns = [
    'Age', 'Sex', 'Race', 'Ethnicity', 'Location', 'Employment', 'Education',
    'Income', 'Health Insurance', 'Relationship to Loved One', 'Years Caring',
    'Daily Caregiving Hours'
]

# =======================
# Load Tokenizer and Config
# =======================
model_ckpt = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = BertTokenizer.from_pretrained(model_ckpt)
config = BertConfig.from_pretrained(model_ckpt, num_labels=2)

# =======================
# LOSO-CV Training
# =======================
subject_ids = data_df['ID'].unique()
all_preds, all_labels = [], []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for test_id in tqdm(subject_ids, desc="LOSO-CV"):
    train_df = data_df[data_df['ID'] != test_id].copy()
    test_df = data_df[data_df['ID'] == test_id].copy()

    train_encodings = tokenizer(list(train_df[text_column]), truncation=True, padding=True, max_length=256)
    test_encodings = tokenizer(list(test_df[text_column]), truncation=True, padding=True, max_length=256)

    train_dataset = TextTabularDataset(train_encodings, train_df[tabular_columns], train_df[target_column])
    test_dataset = TextTabularDataset(test_encodings, test_df[tabular_columns], test_df[target_column])

    model = BertWithTabular(config, tabular_dim=len(tabular_columns))
    model.bert = BertModel.from_pretrained(model_ckpt)
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
        report_to="none",
        lr_scheduler_type="constant",
        warmup_steps=0
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
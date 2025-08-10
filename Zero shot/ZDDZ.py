from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
from sklearn.metrics import f1_score
from tqdm import tqdm
import re
from huggingface_hub import login

login(token="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")

model_name = "deepseek-ai/deepseek-llm-7b-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)

import random
import numpy as np
import torch


# Load CSV
df = pd.read_csv("ZBI_preprocessed.csv")
participant_id_col = "Participant ID"
target_col = "ZBI_score"
feature_cols = [col for col in df.columns if col not in [participant_id_col, target_col]]

all_preds = []
all_labels = []

subject_ids = df[participant_id_col].unique()

for test_id in tqdm(subject_ids, desc="LOSO-CV"):
    test_df = df[df[participant_id_col] == test_id]

    for _, row in test_df.iterrows():
        test_example = ", ".join([f"{col}: {row[col]}" for col in feature_cols])
        prompt = (
            "You are a helpful assistant that predicts high burden using the input features."
            " Given the features below, output only the digit '0' or '1' to predict high burden with no explanation."
            "\n\n1 means the participant is at risk of high burden."
            "\n0 means the participant is not at risk of high burden."
            "\n\nImportant:"
            "\n- Only answer with the digit 0 or 1."
            "\n- Do not include any text or explanation."
            f"\n\nFeatures: {test_example}\n\nAnswer:"
        )

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=5, temperature=0)

        # Extract generated text excluding the prompt
        generated_tokens = outputs[0][inputs['input_ids'].shape[-1]:]
        decoded_output = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        print(f"Decoded output: '{decoded_output}'")

        # Extract prediction (0 or 1)
        match = re.search(r'\b(0|1)\b', decoded_output)
        prediction = int(match.group(1)) if match else 0

        all_preds.append(prediction)
        all_labels.append(row["ZBI_score"])

# Compute F1-score
final_f1 = f1_score(all_labels, all_preds, average="binary")
print(f"\nFinal LOSO-CV F1 Score: {final_f1:.4f}")
print("Predictions:", all_preds)
print("Labels:", all_labels)

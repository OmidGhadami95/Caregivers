from huggingface_hub import login
login(token="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")

model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)

import random
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import re
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForCausalLM


# Load CSVs
df = pd.read_csv("ZBI_preprocessed.csv")
narrative_df = pd.read_csv("preprocessed_narrative.csv")

# Merge UNI feature into the main dataframe
df = pd.merge(df, narrative_df[['ID', 'UNI']], left_on='Participant ID', right_on='ID', how='left')

# Define participant ID, target, and feature columns
participant_id_col = "Participant ID"
target_col = "ZBI_score"
feature_cols = [col for col in df.columns if col not in [participant_id_col, target_col, 'ID']]

all_preds = []
all_labels = []

# LOSO-CV loop
subject_ids = df[participant_id_col].unique()

for test_id in tqdm(subject_ids, desc="LOSO-CV"):
    test_df = df[df[participant_id_col] == test_id]

    for _, row in test_df.iterrows():
        test_example = ", ".join([f"{col}: {row[col]}" for col in feature_cols])
        prompt = (
            "You are a highly reliable assistant that predicts high durden risk using the ZBI score."
            " Based on input features, You must output only a single digit: 0 or 1, to predict high burden with NO text, NO explanation, and NO punctuation."
            "\n\n1 means the participant is at risk of high burden."
            "\n0 means the participant is not at risk of high burden."
            "\n\nImportant:"
            "\n- Output ONLY the digit 0 or 1."
            "\n- Absolutely NO text, punctuation, or explanation."
            "\n\n### Features:\n"
            f"{test_example}"
            "\n\n### Answer:\n"
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

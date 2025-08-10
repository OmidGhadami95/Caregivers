import pandas as pd
import re
from sklearn.metrics import f1_score
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
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
        prompt =[{"role": "system", "content": "You are a helpful assistant that predicts high burden risk using the features."},
                 {"role": "user", "content": "Below, are demographic features of a person. Given these features, classify this person as a high burden risk (0) or a low burden risk (1). FINAL ANSWER FORMATTING INSTRUCTION: Your final answer MUST be a single letter, in the form ‘/boxed[0]/’ or ‘/boxed[1]/’, at the end of your response.FEATURES:\n"}]
        print(test_example)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = pipeline(messages,
            max_new_tokens= 1000,
	    do_sample= False,
	    num_beams= 1,
	    temperature= 0,
	    top_k= None,
	    top_p= 1.0)




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

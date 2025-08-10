from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
from sklearn.metrics import f1_score
from tqdm import tqdm
import re
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

# Set random seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load narrative CSV
narrative_df = pd.read_csv("preprocessed_narrative.csv")

# Define participant ID, target, and feature columns
participant_id_col = "ID"
target_col = "Depressed"  # or replace with the appropriate target column if different
text_column = "UNI"

all_preds = []
all_labels = []

# LOSO-CV loop
subject_ids = narrative_df[participant_id_col].unique()

for test_id in tqdm(subject_ids, desc="LOSO-CV"):
    test_df = narrative_df[narrative_df[participant_id_col] == test_id]

    for _, row in test_df.iterrows():
        narrative_text = row[text_column]
        prompt = (
            "You are a helpful assistant that predicts depression risk using the input."
            " Based on the narrative below, output only the digit '0' or '1' with no explanation to estimate depression."
            "\n\n1 means the participant is at risk of depression."
            "\n0 means the participant is not at risk of depression."
            "\n\nImportant:"
            "\n- Only answer with the digit 0 or 1."
            "\n- Do not include any text or explanation."
            f"\n\input: {narrative_text}\n\nAnswer:"
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
        all_labels.append(row[target_col])

# Compute F1-score
final_f1 = f1_score(all_labels, all_preds, average="binary")
print(f"\nFinal LOSO-CV F1 Score: {final_f1:.4f}")
print("Predictions:", all_preds)
print("Labels:", all_labels)

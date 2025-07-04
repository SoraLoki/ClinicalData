import pandas as pd
import numpy as np
import torch
import ast
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from transformers import BertTokenizerFast, BertModel

# --- Global Configuration ---
LABEL_LIST = ['O', 'B-FEATURE', 'I-FEATURE']
LABEL_MAP = {label: i for i, label in enumerate(LABEL_LIST)}
IGNORE_LABEL_ID = -100
MAX_LEN = 512
EPOCHS = 3
LOSS_LOG_PATH = "loss_log.json"  # Datei zur Speicherung der Loss-Werte pro Epoche
BATCH_SIZE = 24
LEARNING_RATE = 2e-5
DATA_PATH = "nbme-score-clinical-patient-notes/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Dataset Class ---
class PatientNotesTokenClassificationDataset(Dataset):
    def __init__(self, encodings_list, labels_list):
        self.encodings_list = encodings_list
        self.labels_list = labels_list

    def __getitem__(self, idx):
        item = self.encodings_list[idx]
        item['labels'] = self.labels_list[idx]
        return item

    def __len__(self):
        return len(self.labels_list)

# --- Preprocessing Function ---
def preprocess_data_for_token_classification(df, tokenizer, max_len):
    all_input_ids = []
    all_attention_masks = []
    all_labels = []
    all_offset_mappings = []
    all_sequence_ids = []
    all_contexts = []

    for _, row in df.iterrows():
        question = str(row['feature_text'])
        context = str(row['pn_history'])
        location_str = str(row['location'])

        encoding = tokenizer.encode_plus(
            question, context,
            add_special_tokens=True,
            max_length=max_len,
            padding="max_length",
            truncation="only_second",
            return_offsets_mapping=True,
            return_attention_mask=True
        )

        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        offset_mapping = encoding['offset_mapping']
        sequence_ids = encoding.sequence_ids()

        token_labels = [IGNORE_LABEL_ID] * len(input_ids)
        for i in range(len(input_ids)):
            if sequence_ids[i] == 1:
                token_labels[i] = LABEL_MAP['O']

        try:
            char_spans = ast.literal_eval(location_str)
        except:
            char_spans = []

        parsed_spans = []
        for span_str in char_spans:
            try:
                start_char, end_char = map(int, span_str.split())
                if start_char < end_char:
                    parsed_spans.append((start_char, end_char))
            except:
                continue

        for start_char, end_char in parsed_spans:
            first = True
            for token_idx in range(len(input_ids)):
                if sequence_ids[token_idx] != 1:
                    continue
                token_start, token_end = offset_mapping[token_idx]
                if max(token_start, start_char) < min(token_end, end_char):
                    if first:
                        token_labels[token_idx] = LABEL_MAP['B-FEATURE']
                        first = False
                    else:
                        if token_labels[token_idx] != LABEL_MAP['B-FEATURE']:
                            token_labels[token_idx] = LABEL_MAP['I-FEATURE']

        all_input_ids.append(torch.tensor(input_ids))
        all_attention_masks.append(torch.tensor(attention_mask))
        all_labels.append(torch.tensor(token_labels))
        all_offset_mappings.append(offset_mapping)
        all_sequence_ids.append(sequence_ids)
        all_contexts.append(context)

    encodings_list = []
    for i in range(len(all_input_ids)):
        encodings_list.append({
            'input_ids': all_input_ids[i],
            'attention_mask': all_attention_masks[i]
        })

    eval_info = {
        'offset_mapping': all_offset_mappings,
        'sequence_ids': all_sequence_ids,
        'contexts': all_contexts,
        'original_df': df.copy()
    }

    return encodings_list, all_labels, eval_info

def log_loss_to_file(epoch, train_loss, val_loss, filepath=LOSS_LOG_PATH):
    import os
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
    else:
        data = []

    data.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss": val_loss
    })

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

# --- Span Utilities ---
def get_char_sets_from_location_str(location_str, pn_text_len):
    char_sets = []
    try:
        char_spans = ast.literal_eval(location_str)
        for span_str in char_spans:
            parts = span_str.split()
            if len(parts) == 2:
                start, end = int(parts[0]), int(parts[1])
                start = max(0, start)
                end = min(pn_text_len, end)
                if start < end:
                    char_sets.append(set(range(start, end)))
    except:
        pass
    return char_sets

# --- Attention-based Evaluation ---
def evaluate_attention_based_spans(model, tokenizer, df, device, max_len, topk=5):
    model.eval()
    all_jaccards = []
    all_predicted_span_lengths = []
    all_true_span_lengths = []


    for i, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating Attention-Based Spans"):
        question = str(row['feature_text'])
        context = str(row['pn_history'])
        location_str = str(row['location'])

        encoding = tokenizer(
            question, context,
            return_tensors="pt",
            return_offsets_mapping=True,
            return_attention_mask=True,
            max_length=max_len,
            truncation="only_second",
            padding="max_length"
        )
        offset_mapping = encoding['offset_mapping'][0].tolist()
        sequence_ids = encoding.sequence_ids(0)

        inputs = {k: v.to(device) for k, v in encoding.items() if k != 'offset_mapping'}

        with torch.no_grad():
            output = model(**inputs, output_attentions=True)

        attentions = output.attentions
        mean_attention = torch.stack(attentions).mean(dim=0).mean(dim=1).squeeze(0)
        cls_attention = mean_attention[0]

        # Top 10 Kontext-Tokens mit höchster Attention
        token_attn_pairs = [
            (idx, attn_val) for idx, (attn_val, sid) in enumerate(zip(cls_attention.tolist(), sequence_ids))
            if sid == 1
        ]
        token_attn_pairs = sorted(token_attn_pairs, key=lambda x: x[1], reverse=True)[:10]
        relevant_tokens = [idx for idx, _ in token_attn_pairs]


        # Wenn nichts relevant ist, skip
        if not relevant_tokens:
            char_spans = []
        else:
            # Clusterbildung nach Nähe (z. B. max 2 Token auseinander)
            clusters = [[relevant_tokens[0]]]
            for current in relevant_tokens[1:]:
                if current - clusters[-1][-1] <= 2:
                    clusters[-1].append(current)
                else:
                    clusters.append([current])

            # In Char-Spans umwandeln
            char_spans = []
            for cluster in clusters:
                start_idx, end_idx = cluster[0], cluster[-1]
                
                delta = 2  # Anzahl zusätzlicher Tokens links/rechts

                # Links ausdehnen
                start_idx_exp = max(0, start_idx - delta)
                # Rechts ausdehnen (bis max. len(offset_mapping) - 1)
                end_idx_exp = min(len(offset_mapping) - 1, end_idx + delta)

                span_start = offset_mapping[start_idx_exp][0]
                span_end = offset_mapping[end_idx_exp][1]

                if span_start < span_end:
                    char_spans.append(set(range(span_start, span_end)))



        predicted_char_set = set().union(*char_spans)
        true_char_sets = get_char_sets_from_location_str(location_str, len(context))
        true_char_set = set().union(*true_char_sets)

        if not predicted_char_set and not true_char_set:
            jaccard = 1.0
        elif not predicted_char_set or not true_char_set:
            jaccard = 0.0
        else:
            intersection = predicted_char_set & true_char_set
            union = predicted_char_set | true_char_set
            jaccard = len(intersection) / len(union)

        all_jaccards.append(jaccard)
        all_predicted_span_lengths.extend([len(span) for span in char_spans])
        all_true_span_lengths.extend([len(span) for span in true_char_sets])


    mean_jaccard = np.mean(all_jaccards)
    print(f"Average Jaccard score (attention-based): {mean_jaccard:.4f}")
    # Speichern der Spanlängen zur Analyse
    span_length_data = {
        "predicted": all_predicted_span_lengths,
        "true": all_true_span_lengths
    }

    with open("span_length_distribution.json", "w") as f:
        json.dump(span_length_data, f, indent=2)

    return mean_jaccard

# --- Load and Merge Data ---
features_df = pd.read_csv(f"{DATA_PATH}features.csv")
notes_df = pd.read_csv(f"{DATA_PATH}patient_notes.csv")
train_df = pd.read_csv(f"{DATA_PATH}train.csv")

merged_df = train_df.merge(features_df, on=['feature_num', 'case_num'], how='left')
merged_df = merged_df.merge(notes_df, on=['pn_num', 'case_num'], how='left')
merged_df.dropna(subset=['pn_history', 'feature_text', 'location'], inplace=True)

model_name = "bert-base-uncased"
tokenizer = BertTokenizerFast.from_pretrained(model_name)
bert_model = BertModel.from_pretrained(model_name, output_attentions=True)
bert_model.to(device)

# --- Split & Preparation ---
merged_df = merged_df.sample(n=500, random_state=42).reset_index(drop=True)
train_data, test_data = train_test_split(merged_df, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

from transformers import BertForTokenClassification

# Modell mit Klassifikationskopf laden (für echtes Training)
model = BertForTokenClassification.from_pretrained(model_name, num_labels=len(LABEL_LIST))
model.to(device)

# Tokenisierung + Vorbereitung
train_encodings, train_labels, _ = preprocess_data_for_token_classification(train_data, tokenizer, MAX_LEN)
val_encodings, val_labels, _ = preprocess_data_for_token_classification(val_data, tokenizer, MAX_LEN)

train_dataset = PatientNotesTokenClassificationDataset(train_encodings, train_labels)
val_dataset = PatientNotesTokenClassificationDataset(val_encodings, val_labels)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# --- Trainingsschleife ---
for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    # --- Validierung ---
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            total_val_loss += outputs.loss.item()

    avg_val_loss = total_val_loss / len(val_loader)

    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
    log_loss_to_file(epoch, avg_train_loss, avg_val_loss)

# --- Jaccard Score mit Attention-Spans nach Training ---
jaccard_score = evaluate_attention_based_spans(model.bert, tokenizer, test_data, device, MAX_LEN)
print("Done.")


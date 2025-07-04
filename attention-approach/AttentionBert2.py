import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast
from transformers import logging as hf_logging
import numpy as np
from tqdm import tqdm
import ast
import json
import pandas as pd
from sklearn.model_selection import train_test_split

hf_logging.set_verbosity_error()

LABEL_LIST = ['O', 'B-FEATURE', 'I-FEATURE']
LABEL_MAP = {label: i for i, label in enumerate(LABEL_LIST)}
IGNORE_LABEL_ID = -100
MAX_LEN = 512
EPOCHS = 3
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
DATA_PATH = "/app/nbme-score-clinical-patient-notes/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class BertForTokenClassificationWithAttentionLoss(nn.Module):
    def __init__(self, model_name, num_labels, alpha=1.0):
        super().__init__()
        from transformers import BertModel
        self.bert = BertModel.from_pretrained(model_name, output_attentions=True)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.alpha = alpha
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            output_attentions=True,
                            return_dict=True)

        sequence_output = outputs.last_hidden_state
        logits = self.classifier(self.dropout(sequence_output))
        output = {'logits': logits}

        if labels is not None:
            loss_cls = self.loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
            cls_attn = outputs.attentions[-1][:, :, 0, :]
            cls_attn_mean = cls_attn.mean(dim=1)
            attn_target = ((labels == 1) | (labels == 2)).float()
            cls_attn_norm = cls_attn_mean / (cls_attn_mean.sum(dim=1, keepdim=True) + 1e-8)
            attention_loss = nn.functional.kl_div(
                cls_attn_norm.log(),
                attn_target / (attn_target.sum(dim=1, keepdim=True) + 1e-8),
                reduction="batchmean"
            )
            total_loss = loss_cls + self.alpha * attention_loss
            output['loss'] = total_loss
            output['loss_cls'] = loss_cls
            output['loss_attn'] = attention_loss

        return output

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

def preprocess_data_for_token_classification(df, tokenizer, max_len):
    all_input_ids = []
    all_attention_masks = []
    all_labels = []

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

    encodings_list = []
    for i in range(len(all_input_ids)):
        encodings_list.append({
            'input_ids': all_input_ids[i],
            'attention_mask': all_attention_masks[i]
        })

    return encodings_list, all_labels

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

def to_serializable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu().tolist()
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    else:
        return obj

def evaluate_attention_based_spans(model, tokenizer, df, device, max_len, topk=5):
    model.eval()
    all_jaccards = []

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
            output = model(**inputs)
            attentions = output['logits']
            cls_attn = model.bert(input_ids=inputs['input_ids'],
                                  attention_mask=inputs['attention_mask'],
                                  output_attentions=True,
                                  return_dict=True).attentions[-1][:, :, 0, :].mean(dim=1).squeeze(0)

        token_attn_pairs = [
            (idx, attn_val) for idx, (attn_val, sid) in enumerate(zip(cls_attn.tolist(), sequence_ids))
            if sid == 1
        ]
        token_attn_pairs = sorted(token_attn_pairs, key=lambda x: x[1], reverse=True)[:topk]
        relevant_tokens = [idx for idx, _ in token_attn_pairs]

        char_spans = []
        if relevant_tokens:
            clusters = [[relevant_tokens[0]]]
            for current in relevant_tokens[1:]:
                if current - clusters[-1][-1] <= 2:
                    clusters[-1].append(current)
                else:
                    clusters.append([current])
            for cluster in clusters:
                start_idx, end_idx = cluster[0], cluster[-1]
                start_idx_exp = max(0, start_idx - 3)
                end_idx_exp = min(len(offset_mapping) - 1, end_idx + 3)
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

    mean_jaccard = np.mean(all_jaccards)
    print(f"Average Jaccard score (attention-based): {mean_jaccard:.4f}")
    return mean_jaccard

def train_model():
    print("Loading data...")
    features_df = pd.read_csv(f"{DATA_PATH}features.csv")
    notes_df = pd.read_csv(f"{DATA_PATH}patient_notes.csv")
    train_df = pd.read_csv(f"{DATA_PATH}train.csv")

    merged_df = train_df.merge(features_df, on=['feature_num', 'case_num'], how='left')
    merged_df = merged_df.merge(notes_df, on=['pn_num', 'case_num'], how='left')
    merged_df.dropna(subset=['pn_history', 'feature_text', 'location'], inplace=True)

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    merged_df = merged_df.reset_index(drop=True)
    train_data, test_data = train_test_split(merged_df, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

    train_encodings, train_labels = preprocess_data_for_token_classification(train_data, tokenizer, MAX_LEN)
    val_encodings, val_labels = preprocess_data_for_token_classification(val_data, tokenizer, MAX_LEN)

    train_dataset = PatientNotesTokenClassificationDataset(train_encodings, train_labels)
    val_dataset = PatientNotesTokenClassificationDataset(val_encodings, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = BertForTokenClassificationWithAttentionLoss("bert-base-uncased", num_labels=len(LABEL_LIST), alpha=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    model.to(device)

    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_cls_loss, train_attn_loss = 0.0, 0.0, 0.0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, labels=labels)
            loss = outputs['loss']
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_cls_loss += outputs['loss_cls'].item()
            train_attn_loss += outputs['loss_attn'].item()

        avg_train_loss = train_loss / len(train_loader)
        avg_cls = train_cls_loss / len(train_loader)
        avg_attn = train_attn_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, CLS: {avg_cls:.4f}, ATTN: {avg_attn:.4f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask, labels=labels)
                val_loss += outputs['loss'].item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} - Val Loss: {avg_val_loss:.4f}")

        print("Evaluating model via attention-based span extraction...")
        attention_jaccard = evaluate_attention_based_spans(
            model, tokenizer, test_data, device, MAX_LEN, topk=5
        )
        print(f"Final Attention-Jaccard Score (Epoch {epoch+1}): {attention_jaccard:.4f}")

if __name__ == '__main__':
    train_model()
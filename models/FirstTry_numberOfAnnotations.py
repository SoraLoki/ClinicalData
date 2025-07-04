import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForQuestionAnswering
import torch.optim as optim
import numpy as np
import ast 

LABEL_LIST = ['O', 'B-FEATURE', 'I-FEATURE']
LABEL_MAP = {label: i for i, label in enumerate(LABEL_LIST)}
IGNORE_LABEL_ID = -100 

def preprocess_data_for_token_classification(df, tokenizer, max_len):
   
    all_input_ids = []
    all_attention_masks = []
    all_labels = []
    for _, row in df.iterrows():
        question = str(row['feature_text'])
        context = str(row['pn_history'])
        location_str = str(row['location'])

        # Tokenization
        encoding = tokenizer.encode_plus(
            question,
            context,
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

        token_labels = [IGNORE_LABEL_ID] * len(input_ids)
        sequence_ids = encoding.sequence_ids()

        for i in range(len(input_ids)):
            if sequence_ids[i] == 1: 
                token_labels[i] = LABEL_MAP['O']
        
        try:
            char_spans = ast.literal_eval(location_str) 
        except (ValueError, SyntaxError):
            char_spans = []

        parsed_char_spans = []
        for span_str in char_spans:
            try:
                start_char, end_char = map(int, span_str.split())
                if start_char < end_char : 
                    parsed_char_spans.append((start_char, end_char))
            except ValueError:
                continue 

        for start_char, end_char in parsed_char_spans:
            first_token_in_current_span = True
            for token_idx in range(len(input_ids)):
                if sequence_ids[token_idx] != 1: 
                    continue

                token_char_start, token_char_end = offset_mapping[token_idx]

                if token_char_start == token_char_end == 0: 
                    continue
                
                if max(token_char_start, start_char) < min(token_char_end, end_char):
                    if first_token_in_current_span:
                        token_labels[token_idx] = LABEL_MAP['B-FEATURE']
                        first_token_in_current_span = False
                    else:
                        # overrite 'O' or 'I-FEATURE'
                        if token_labels[token_idx] != LABEL_MAP['B-FEATURE']:
                             token_labels[token_idx] = LABEL_MAP['I-FEATURE']
        
        all_input_ids.append(torch.tensor(input_ids))
        all_attention_masks.append(torch.tensor(attention_mask))
        all_labels.append(torch.tensor(token_labels))
 
    inputs_list = []
    for i in range(len(all_input_ids)):
        inputs_list.append({
            'input_ids': all_input_ids[i],
            'attention_mask': all_attention_masks[i]
        })
        
    return inputs_list, all_labels 

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

def perform_train_val_test_split(df, train_size=0.8, val_size=0.1, test_size=0.1, random_state=42):
    
    if not (train_size + val_size + test_size == 1.0):
        raise ValueError("The sum of train_size, val_size and test_size must be 1.0.")

    traintemp_df, temp_df = train_test_split(
        df,
        train_size=train_size,
        random_state=random_state
    )

    relative_val_size = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=relative_val_size, 
        random_state=random_state
    )
    return traintemp_df, val_df, test_df

from tqdm import tqdm 

def train_bert_token_classification_model(train_dataloader, val_dataloader, model, optimizer, device, epochs=3):
    
    model.to(device)

    for epoch in range(epochs):

        model.train() 
        total_train_loss = 0
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
    
        train_progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1} Training", leave=False)
        
        for batch_idx, batch in enumerate(train_progress_bar):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                labels=labels 
            )
            
            loss = outputs.loss

            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()

            train_progress_bar.set_postfix({'loss': loss.item()})

        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1} - Average Training Loss: {avg_train_loss:.4f}")

        model.eval() 
        total_val_loss = 0
        
        val_progress_bar = tqdm(val_dataloader, desc=f"Epoch {epoch+1} Validation", leave=False)

        with torch.no_grad(): 
            for batch_idx, batch in enumerate(val_progress_bar):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device) 

                outputs = model(
                    input_ids,
                    attention_mask=attention_mask,
                    labels=labels 
                )
                
                loss = outputs.loss
                total_val_loss += loss.item()
                val_progress_bar.set_postfix({'val_loss': loss.item()})

        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f"Epoch {epoch + 1} - Average Validation Loss: {avg_val_loss:.4f}")
        print("-" * 50)

    print("Training finished.")
    return model

from transformers import BertForTokenClassification
if __name__ == '__main__':
    
    BERT_MODEL_NAME = 'bert-base-uncased'
    MAX_LEN = 512  
    BATCH_SIZE = 8 
    EPOCHS = 5

    model = BertForTokenClassification.from_pretrained(
        BERT_MODEL_NAME,
        num_labels=len(LABEL_LIST)
    )


    DATA_PATH = "nbme-score-clinical-patient-notes/" 
    features_df = pd.read_csv(f"{DATA_PATH}features.csv")
    notes_df = pd.read_csv(f"{DATA_PATH}patient_notes.csv")
    train_df_raw = pd.read_csv(f"{DATA_PATH}train.csv")

    merged_df = train_df_raw.merge(features_df, on=['feature_num', 'case_num'], how='left') \
                            .merge(notes_df, on=['pn_num', 'case_num'], how='left')

    print(f"Size of the merged data set: {merged_df.shape}")
    merged_df.dropna(subset=['pn_history', 'feature_text', 'location'], inplace=True) 

    sample_n = 2000
    sample_df = merged_df.sample(n=sample_n, random_state=42) if len(merged_df) > sample_n else merged_df
    print(f"Verwende Sample von {len(sample_df)} für Training/Validierung.")

    train_data, val_data, test_data_split = perform_train_val_test_split(
        merged_df,
        train_size=0.8,
        val_size=0.1, 
        test_size=0.1,
        random_state=42
    )
    print(f"Size of the train split: {train_data.shape}")
    print(f"Size of the validation split: {val_data.shape}")
    print(f"Size of the test split: {test_data_split.shape}")

    tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL_NAME)

    train_inputs_list, train_labels_list = preprocess_data_for_token_classification(train_data, tokenizer, MAX_LEN)
    train_dataset = PatientNotesTokenClassificationDataset(train_inputs_list, train_labels_list)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_inputs_list, val_labels_list = preprocess_data_for_token_classification(val_data, tokenizer, MAX_LEN)
    val_dataset = PatientNotesTokenClassificationDataset(val_inputs_list, val_labels_list)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    optimizer =optim. AdamW(model.parameters(), lr=5e-5) 

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("NVIDIA GPU (CUDA) used")
    else:
        device = torch.device("cpu")
        print("CPU used")

    model.to(device) 

    print("\nStarte Training für Token Classification...")
    trained_model = train_bert_token_classification_model(
        train_dataloader,
        val_dataloader,
        model,
        optimizer,
        device,
        epochs=EPOCHS
    )

    print("Token Classification Training abgeschlossen.")
    
import ast


def get_char_sets_from_location_str(location_str: str, pn_text_len: int) -> list[set[int]]:
    
    true_spans_char_sets = []
    try:
        char_spans_from_str = ast.literal_eval(location_str)
        for span_str in char_spans_from_str:
            parts = span_str.split()
            if len(parts) == 2:
                start_char, end_char = int(parts[0]), int(parts[1])
                start_char = max(0, start_char)
                end_char = min(pn_text_len, end_char) 
                if start_char < end_char: 
                    true_spans_char_sets.append(set(range(start_char, end_char)))
    except (ValueError, SyntaxError):
        pass
    
    return true_spans_char_sets


def calculate_overall_example_jaccard(
    predicted_char_sets: list[set[int]], 
    true_char_sets: list[set[int]]
) -> float:

    union_predicted_chars = set()
    for pred_set in predicted_char_sets:
        union_predicted_chars.update(pred_set)
        
    union_true_chars = set()
    for true_set in true_char_sets:
        union_true_chars.update(true_set)

    if not union_predicted_chars and not union_true_chars:
        return 1.0
    if not union_predicted_chars or not union_true_chars:
        return 0.0

    intersection_of_unions = len(union_predicted_chars.intersection(union_true_chars))
    union_of_unions = len(union_predicted_chars.union(union_true_chars))

    if union_of_unions == 0:
        return 1.0 if intersection_of_unions == 0 else 0.0 
    
    return intersection_of_unions / union_of_unions


import torch
import numpy as np
from tqdm import tqdm 
import ast 


def extract_char_spans_from_token_labels(
    token_label_ids: list[int], 
    offset_mapping: list[tuple[int, int]], 
    sequence_ids: list[int | None],
    label_map: dict[str, int]
) -> list[set[int]]:
    
    predicted_char_sets = []
    active_span_tokens = [] 

    b_feature_id = label_map.get('B-FEATURE')
    i_feature_id = label_map.get('I-FEATURE')

    for i, label_id in enumerate(token_label_ids):
        current_token_is_context = (sequence_ids[i] == 1)
        token_char_start, token_char_end = offset_mapping[i]
        
        is_valid_context_token = current_token_is_context and not (token_char_start == 0 and token_char_end == 0)

        if label_id == b_feature_id and is_valid_context_token:

            if active_span_tokens:
                span_start_char = active_span_tokens[0][0]
                span_end_char = active_span_tokens[-1][1]
                if span_start_char < span_end_char:
                    predicted_char_sets.append(set(range(span_start_char, span_end_char)))
            
            active_span_tokens = [(token_char_start, token_char_end)]
        elif label_id == i_feature_id and is_valid_context_token:
            if active_span_tokens: 
                active_span_tokens.append((token_char_start, token_char_end))

        else: 
            if active_span_tokens:
                span_start_char = active_span_tokens[0][0]
                span_end_char = active_span_tokens[-1][1]
                if span_start_char < span_end_char:
                    predicted_char_sets.append(set(range(span_start_char, span_end_char)))
                active_span_tokens = [] 
    
    if active_span_tokens:
        span_start_char = active_span_tokens[0][0]
        span_end_char = active_span_tokens[-1][1]
        if span_start_char < span_end_char:
            predicted_char_sets.append(set(range(span_start_char, span_end_char)))
            
    return predicted_char_sets


def evaluate_token_classification_model(model, tokenizer, df_val, device, max_len, label_map):
   
    model.eval() 
    all_jaccards = []
    
    for _, row in tqdm(df_val.iterrows(), total=df_val.shape[0], desc="Validating"):
        question = str(row['feature_text'])
        context = str(row['pn_history'])
        true_location_str = str(row['location'])
        
        inputs = tokenizer.encode_plus(
            question,
            context,
            add_special_tokens=True,
            max_length=max_len,
            padding="max_length",
            truncation="only_second",
            return_offsets_mapping=True,
            return_attention_mask=True, 
            return_tensors="pt"
        )
        
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
       
        offset_mapping_list = inputs['offset_mapping'].squeeze().tolist()
        sequence_ids_list = inputs.sequence_ids(0) 

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits 
            predicted_label_ids = torch.argmax(logits, dim=2).squeeze().tolist() 

        predicted_char_sets_list = extract_char_spans_from_token_labels(
            predicted_label_ids,
            offset_mapping_list,
            sequence_ids_list,
            label_map 
        )
        
        true_char_sets_list = get_char_sets_from_location_str(true_location_str, len(context))
        
        jaccard_score = calculate_overall_example_jaccard(predicted_char_sets_list, true_char_sets_list)
        all_jaccards.append(jaccard_score)
        
    mean_jaccard = np.mean(all_jaccards) if all_jaccards else 0.0
    print(f"Average Jaccard score on the validation dataset {mean_jaccard:.4f}")
    return mean_jaccard

trained_model.to(device) 

print("\nStart evaluation on the validation dataset (Token Classification)...")

average_jaccard = evaluate_token_classification_model( 
    trained_model, 
    tokenizer, 
    val_data,      
    device, 
    MAX_LEN,
    LABEL_MAP     
)

import pandas as pd
import ast 
from collections import Counter

DATA_PATH = "nbme-score-clinical-patient-notes/" 
try:
    train_df = pd.read_csv(f"{DATA_PATH}train.csv")
except FileNotFoundError:
    print(f"Fehler: Die Datei 'train.csv' wurde nicht im Pfad '{DATA_PATH}' gefunden.")
    exit()

print(f"Trainingsdatensatz (train.csv) geladen mit {train_df.shape[0]} Zeilen.")

num_spans_per_row = []

for index, row in train_df.iterrows():
    location_str = str(row['location'])
    list_of_span_strings = ast.literal_eval(location_str)
    num_spans_per_row.append(len(list_of_span_strings))
    
span_counts = Counter(num_spans_per_row)

count_0_annotations = span_counts.get(0, 0)
count_1_annotation = span_counts.get(1, 0)
count_2_annotations = span_counts.get(2, 0)
count_3_annotations = span_counts.get(3, 0)
count_more_than_3_annotations = 0

for num, count in span_counts.items():
    if num >= 4:
        count_more_than_3_annotations += count

total_examples_processed = len(num_spans_per_row)

print("\n--- Verteilung der Anzahl von Annotationen (Spannen) pro Beispiel ---")
print(f"Examples with 0 annotations: {count_0_annotations} ({count_0_annotations/total_examples_processed:.2%})")
print(f"Examples with 1 annotations:  {count_1_annotation} ({count_1_annotation/total_examples_processed:.2%})")
print(f"Examples with 2 annotations {count_2_annotations} ({count_2_annotations/total_examples_processed:.2%})")
print(f"Examples with 3 annotations: {count_3_annotations} ({count_3_annotations/total_examples_processed:.2%})")
print(f"Examples with 4+ annotations: {count_more_than_3_annotations} ({count_more_than_3_annotations/total_examples_processed:.2%})")
print("--------------------------------------------------------------------")
print(f"Total count of annotations: {total_examples_processed}")

print("\Detailled Counts (with more than 3):")
for num, count in sorted(span_counts.items()):
    if num > 3:
        print(f"  Examples with {num} annotations: {count}")


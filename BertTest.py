import pandas as pd
import numpy as np
import torch
import ast
import itertools 
from tqdm import tqdm
from sklearn.model_selection import train_test_split, KFold 
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForTokenClassification


# --- Global Configuration & Label Definitions ---
LABEL_LIST = ['O', 'B-FEATURE', 'I-FEATURE']
LABEL_MAP = {label: i for i, label in enumerate(LABEL_LIST)}
IGNORE_LABEL_ID = -100

# --- Data Preprocessing (unchanged from original) ---
def preprocess_data_for_token_classification(df, tokenizer, max_len):
    all_input_ids = []
    all_attention_masks = []
    all_labels = []
    all_offset_mappings_for_eval = []
    all_sequence_ids_for_eval = []
    all_contexts_for_eval = []

    for _, row in df.iterrows():
        question = str(row['feature_text'])
        context = str(row['pn_history'])
        location_str = str(row['location'])

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
        sequence_ids = encoding.sequence_ids()

        token_labels = [IGNORE_LABEL_ID] * len(input_ids)
        
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
                        if token_labels[token_idx] != LABEL_MAP['B-FEATURE']:
                             token_labels[token_idx] = LABEL_MAP['I-FEATURE']
        
        all_input_ids.append(torch.tensor(input_ids))
        all_attention_masks.append(torch.tensor(attention_mask))
        all_labels.append(torch.tensor(token_labels))
        all_offset_mappings_for_eval.append(offset_mapping)
        all_sequence_ids_for_eval.append(sequence_ids)
        all_contexts_for_eval.append(context)

    inputs_list = []
    for i in range(len(all_input_ids)):
        inputs_list.append({
            'input_ids': all_input_ids[i],
            'attention_mask': all_attention_masks[i]
        })
    
    eval_info = {
        'offset_mapping': all_offset_mappings_for_eval,
        'sequence_ids': all_sequence_ids_for_eval,
        'contexts': all_contexts_for_eval,
        'original_df': df.copy()
    }
        
    return inputs_list, all_labels, eval_info


# --- PyTorch Dataset Class (unchanged from original) ---
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

# --- NEW: Function to create model and tokenizer ---
def create_model_and_tokenizer(model_name: str, num_labels: int):
    """
    Loads a pretrained model and its tokenizer using the exact classes from the original script.
    """
    print(f"Loading model and tokenizer for: {model_name}")
    model = BertForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    return model, tokenizer

# --- Training & Validation Logic (slightly adapted for modularity) ---
def train_and_validate_epoch(model, train_dataloader, val_dataloader, optimizer, device, epoch_num, total_epochs):
    model.train()
    total_train_loss = 0
    train_progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch_num+1}/{total_epochs} Training", leave=False)

    for batch in train_progress_bar:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_train_loss += loss.item()
        loss.backward()
        optimizer.step()
        train_progress_bar.set_postfix({'loss': loss.item()})

    avg_train_loss = total_train_loss / len(train_dataloader)
    print(f"Epoch {epoch_num+1} - Avg. Training Loss: {avg_train_loss:.4f}")

    model.eval()
    total_val_loss = 0
    val_progress_bar = tqdm(val_dataloader, desc=f"Epoch {epoch_num+1}/{total_epochs} Validation", leave=False)
    with torch.no_grad():
        for batch in val_progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_val_loss += loss.item()
            val_progress_bar.set_postfix({'val_loss': loss.item()})
    
    avg_val_loss = total_val_loss / len(val_dataloader)
    print(f"Epoch {epoch_num+1} - Avg. Validation Loss: {avg_val_loss:.4f}")
    return avg_val_loss

# --- Evaluation Logic (adapted for flexible use) ---
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

def calculate_overall_example_jaccard(predicted_char_sets: list[set[int]], true_char_sets: list[set[int]]) -> float:
    union_predicted_chars = set().union(*predicted_char_sets)
    union_true_chars = set().union(*true_char_sets)
    if not union_predicted_chars and not union_true_chars: return 1.0
    if not union_predicted_chars or not union_true_chars: return 0.0
    intersection_of_unions = len(union_predicted_chars.intersection(union_true_chars))
    union_of_unions = len(union_predicted_chars.union(union_true_chars))
    return intersection_of_unions / union_of_unions if union_of_unions > 0 else 1.0

def extract_char_spans_from_token_labels(token_label_ids, offset_mapping, sequence_ids, label_map):
    predicted_char_sets = []
    active_span_tokens = []
    b_feature_id = label_map.get('B-FEATURE')
    i_feature_id = label_map.get('I-FEATURE')

    for i, label_id in enumerate(token_label_ids):
        is_context = sequence_ids and sequence_ids[i] == 1
        if not is_context:
            if active_span_tokens:
                start, end = active_span_tokens[0][0], active_span_tokens[-1][1]
                if start < end: predicted_char_sets.append(set(range(start, end)))
                active_span_tokens = []
            continue
        
        token_char_start, token_char_end = offset_mapping[i]
        is_valid_token = not (token_char_start == 0 and token_char_end == 0)

        if label_id == b_feature_id and is_valid_token:
            if active_span_tokens:
                start, end = active_span_tokens[0][0], active_span_tokens[-1][1]
                if start < end: predicted_char_sets.append(set(range(start, end)))
            active_span_tokens = [(token_char_start, token_char_end)]
        elif label_id == i_feature_id and is_valid_token and active_span_tokens:
            active_span_tokens.append((token_char_start, token_char_end))
        else: # O-label or other label
            if active_span_tokens:
                start, end = active_span_tokens[0][0], active_span_tokens[-1][1]
                if start < end: predicted_char_sets.append(set(range(start, end)))
                active_span_tokens = []
    
    if active_span_tokens:
        start, end = active_span_tokens[0][0], active_span_tokens[-1][1]
        if start < end: predicted_char_sets.append(set(range(start, end)))
            
    return predicted_char_sets

def evaluate_model_on_data(model, tokenizer, df_to_eval, device, max_len, label_map):
    model.eval()
    all_jaccards = []

    # Preprocess data once
    _, _, eval_info = preprocess_data_for_token_classification(df_to_eval, tokenizer, max_len)
    
    for i in tqdm(range(len(df_to_eval)), desc="Evaluating"):
        row = eval_info['original_df'].iloc[i]
        question = str(row['feature_text'])
        context = str(row['pn_history'])
        true_location_str = str(row['location'])

        inputs = tokenizer.encode_plus(
            question, context,
            max_length=max_len, padding="max_length", truncation="only_second", return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_label_ids = torch.argmax(outputs.logits, dim=2).squeeze().tolist()
            
        predicted_char_sets = extract_char_spans_from_token_labels(
            predicted_label_ids, eval_info['offset_mapping'][i], eval_info['sequence_ids'][i], label_map
        )
        true_char_sets = get_char_sets_from_location_str(true_location_str, len(eval_info['contexts'][i]))
        jaccard_score = calculate_overall_example_jaccard(predicted_char_sets, true_char_sets)
        all_jaccards.append(jaccard_score)

    mean_jaccard = np.mean(all_jaccards)
    print(f"Average Jaccard score on evaluation data: {mean_jaccard:.4f}")
    return mean_jaccard

# --- NOTE: hyperparameter_tuning_with_cv will NOT be used in the modified main block ---
# You can optionally remove this function definition if it's no longer needed at all.
# If you keep it, it will just be an unused function.
def hyperparameter_tuning_with_cv(df, model_name, param_grid, n_splits, max_len, epochs, device):
    """
    Performs a grid search with k-fold cross-validation to find the best hyperparameters.
    (This function is kept for reference but not called in the modified main)
    """
    # ... (function body as before) ...
    pass # Placeholder if you don't want to copy the whole body

# --- Main execution block (Orchestrator) ---
if __name__ == '__main__':


    pn_test = "The patient has chronic heart failure and dyspnea."

    tokenizer_bert = BertTokenizerFast.from_pretrained("bert-base-uncased")
    tokenizer_bluebert = BertTokenizerFast.from_pretrained("JosephNguyen/new-BlueBERT-mimic")

    enc_bert = tokenizer_bert(pn_test, return_offsets_mapping=True)
    enc_blue = tokenizer_bluebert(pn_test, return_offsets_mapping=True)

    print("BERT Tokens:", tokenizer_bert.convert_ids_to_tokens(enc_bert["input_ids"]))
    print("Offsets:", enc_bert["offset_mapping"])

    print("BlueBERT Tokens:", tokenizer_bluebert.convert_ids_to_tokens(enc_blue["input_ids"]))
    print("Offsets:", enc_blue["offset_mapping"])


    # --- Global configuration for the run ---
    DATA_PATH = "nbme-score-clinical-patient-notes/"
    MAX_LEN = 512
    EPOCHS = 3          
    
    # --- Manually set desired hyperparameters (since no CV is performed) ---
    # Sie können diese Werte nach Belieben anpassen.
    FIXED_LEARNING_RATE = 2e-5 
    FIXED_BATCH_SIZE = 24    
    
    # --- Device setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- Load and prepare dataset (use FULL merged_df) ---
    try:
        features_df = pd.read_csv(f"{DATA_PATH}features.csv")
        notes_df = pd.read_csv(f"{DATA_PATH}patient_notes.csv")
        train_df_raw = pd.read_csv(f"{DATA_PATH}train.csv")
    except FileNotFoundError as e:
        print(f"Error: Data not found. Please ensure the path '{DATA_PATH}' is correct. Original error: {e}")
        exit()

    merged_df = train_df_raw.merge(features_df, on=['feature_num', 'case_num'], how='left') \
                            .merge(notes_df, on=['pn_num', 'case_num'], how='left')
    merged_df.dropna(subset=['pn_history', 'feature_text', 'location'], inplace=True)
    
    # Verwende JETZT den gesamten merged_df, keine Stichprobe mehr!
    df_to_use = merged_df.copy() # Eine Kopie erstellen, um sicherzustellen, dass das Original unberührt bleibt
    print(f"\nSize of dataset being used: {len(df_to_use)} entries (Full Dataset)")

    # --- Define models for evaluation ---
    models_to_evaluate = {
	'BioFormer': 'bioformers/bioformer-16L', # Hier anpassen um andere Berts/Modelle zu testen
    'BERT-base-uncased': 'bert-base-uncased',
    }
    
    final_results = {}

    # --- Loop through all models to train and evaluate ---
    for friendly_name, model_name in models_to_evaluate.items():
        print(f"\n{'='*20} TRAINING AND EVALUATING MODEL: {friendly_name.upper()} {'='*20}")
        
        # 1. Daten in Training, Validierung und Test aufteilen (vom gesamten merged_df)
        # 80% train_val_data, 20% test_data
        train_val_data, test_data = train_test_split(df_to_use, test_size=0.2, random_state=42)
        # Von train_val_data: 90% train_data, 10% val_data
        train_data, val_data = train_test_split(train_val_data, test_size=0.1, random_state=42)
        
        print(f"Training data size: {len(train_data)}")
        print(f"Validation data size: {len(val_data)}")
        print(f"Test data size: {len(test_data)}")

        # 2. Modell und Tokenizer erstellen
        current_model, current_tokenizer = create_model_and_tokenizer(model_name, len(LABEL_LIST))
        current_model.to(device)

        # 3. Daten vorbereiten und DataLoader erstellen
        print("Preprocessing training data...")
        train_inputs, train_labels, _ = preprocess_data_for_token_classification(train_data, current_tokenizer, MAX_LEN)
        train_dataset = PatientNotesTokenClassificationDataset(train_inputs, train_labels)
        train_dataloader = DataLoader(train_dataset, batch_size=FIXED_BATCH_SIZE, shuffle=True)

        print("Preprocessing validation data...")
        val_inputs, val_labels, _ = preprocess_data_for_token_classification(val_data, current_tokenizer, MAX_LEN)
        val_dataset = PatientNotesTokenClassificationDataset(val_inputs, val_labels)
        val_dataloader = DataLoader(val_dataset, batch_size=FIXED_BATCH_SIZE)

        # 4. Optimierer initialisieren
        optimizer = optim.AdamW(current_model.parameters(), lr=FIXED_LEARNING_RATE)

        # 5. Modell trainieren
        print(f"\n--- Training {friendly_name} for {EPOCHS} epochs ---")
        for epoch in range(EPOCHS):
            train_and_validate_epoch(current_model, train_dataloader, val_dataloader, optimizer, device, epoch, EPOCHS)

        # 6. Finales Modell auf dem Held-out Testset bewerten
        print(f"\n--- Final evaluation of {friendly_name} on the test set ---")
        test_jaccard = evaluate_model_on_data(current_model, current_tokenizer, test_data, device, MAX_LEN, LABEL_MAP)

        final_results[friendly_name] = {
            'learning_rate_used': FIXED_LEARNING_RATE,
            'batch_size_used': FIXED_BATCH_SIZE,
            'final_test_jaccard': test_jaccard
        }

    # --- Final Summary ---
    print(f"\n\n{'='*25} FINAL RESULTS {'='*25}")
    for model_name, result_data in final_results.items():
        print(f"Model: {model_name}")
        print(f"  - Learning Rate: {result_data['learning_rate_used']}")
        print(f"  - Batch Size: {result_data['batch_size_used']}")
        print(f"  - Jaccard on Final Test Set: {result_data['final_test_jaccard']:.4f}")
        print("-" * 65)

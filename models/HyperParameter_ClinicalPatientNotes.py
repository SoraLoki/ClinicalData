import pandas as pd
import numpy as np
import torch
import ast
import itertools 
from tqdm import tqdm
from sklearn.model_selection import train_test_split, KFold 
import torch.optim as optim 
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer, BertForTokenClassification

# Config & Label definition
LABEL_LIST = ['O', 'B-FEATURE', 'I-FEATURE']
LABEL_MAP = {label: i for i, label in enumerate(LABEL_LIST)}
# Standard ignore index for PyTorch's CrossEntropyLoss
IGNORE_LABEL_ID = -100

# Data Preprocessing
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


# PyTorch Dataset Class
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

# Creating model and tokenizer 
def create_model_and_tokenizer(model_name: str, num_labels: int):
    """
    Loads a pretrained model and its tokenizer using the exact classes from the original script.
    """
    print(f"Loading model and tokenizer for: {model_name}")
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
    # A "Fast" tokenizer is important for the offset_mapping
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    return model, tokenizer

# Training and validation methods
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

# Evaluation logic
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
        else: 
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

# Hyperparameter & Cross Validation 
def hyperparameter_tuning_with_cv(df, model_name, param_grid, n_splits, max_len, epochs, device):
    """
    Performs a grid search with k-fold cross-validation to find the best hyperparameters.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"\nStarting cross-validation for {model_name} with {len(param_combinations)} parameter combinations.")
    
    best_score = -1
    best_params = None

    for params in param_combinations:
        print(f"\n--- Testing parameters: {params} ---")
        current_lr = params['learning_rate']
        current_batch_size = params['batch_size']
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
            print(f"--- Fold {fold + 1}/{n_splits} ---")
            train_data = df.iloc[train_idx]
            val_data = df.iloc[val_idx]

            model, tokenizer = create_model_and_tokenizer(model_name, len(LABEL_LIST))
            model.to(device)

            train_inputs, train_labels, _ = preprocess_data_for_token_classification(train_data, tokenizer, max_len)
            train_dataset = PatientNotesTokenClassificationDataset(train_inputs, train_labels)
            train_dataloader = DataLoader(train_dataset, batch_size=current_batch_size, shuffle=True)

            val_inputs, val_labels, _ = preprocess_data_for_token_classification(val_data, tokenizer, max_len)
            val_dataset = PatientNotesTokenClassificationDataset(val_inputs, val_labels)
            val_dataloader = DataLoader(val_dataset, batch_size=current_batch_size)

            optimizer = optim.AdamW(model.parameters(), lr=current_lr)
            
            for epoch in range(epochs):
                train_and_validate_epoch(model, train_dataloader, val_dataloader, optimizer, device, epoch, epochs)
            
            jaccard_score = evaluate_model_on_data(model, tokenizer, val_data, device, max_len, LABEL_MAP)
            fold_scores.append(jaccard_score)

        avg_cv_score = np.mean(fold_scores)
        print(f"Average Jaccard score for parameters {params}: {avg_cv_score:.4f}")

        if avg_cv_score > best_score:
            best_score = avg_cv_score
            best_params = params
    
    print(f"\nBest Jaccard score in cross-validation: {best_score:.4f}")
    print(f"Best parameters found: {best_params}")
    return best_params, best_score

# Main execution block 
if __name__ == '__main__':
   
    DATA_PATH = "nbme-score-clinical-patient-notes/"
    MAX_LEN = 350
    EPOCHS = 7          
    N_SPLITS_CV = 3     
    SAMPLE_N = 2000     
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load and prepare dataset
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
    
    # Just use a sub sample for faster training
    sample_df = merged_df.sample(n=SAMPLE_N, random_state=42) if len(merged_df) > SAMPLE_N else merged_df
    print(f"\nSize of dataset being used: {len(sample_df)} entries")

    # Defining model and hyperparameter 
    models_to_evaluate = {
	'BioFormer': 'bioformers/BioFormer-16L',
    'BERT-base-uncased': 'bert-base-uncased',
	'Roberta': 'FacebookAI/roberta-large',
    'Clinical Bert': 'AKHIL001/Bio_Clinical_BERT',
    'DeBerta': 'MoritzLaurer/deberta-v3-large-zeroshot-v2.0'
    }
    
    # Grid of hyperparameters 
    param_grid = {
        'learning_rate': [1e-5, 3e-5, 5e-5,5e-5],
        'batch_size': [1,2,4,8,16,24]
    }

    final_results = {}

    # Loop for evaluation 
    for friendly_name, model_name in models_to_evaluate.items():
        print(f"\n{'='*20} EVALUATING MODEL: {friendly_name.upper()} {'='*20}")
        
        best_hyperparams, best_cv_score = hyperparameter_tuning_with_cv(
            df=sample_df,
            model_name=model_name,
            param_grid=param_grid,
            n_splits=N_SPLITS_CV,
            max_len=MAX_LEN,
            epochs=EPOCHS,
            device=device
        )
        
        # Train final model with the whole dataset and the best parameters 
        print(f"\n--- Training final model for {friendly_name} with best params: {best_hyperparams} ---")
        
        # Data split for final training and testing
        train_val_data, test_data = train_test_split(merged_df, test_size=0.2, random_state=42)
        train_data, val_data = train_test_split(train_val_data, test_size=0.1, random_state=42)

        final_model, final_tokenizer = create_model_and_tokenizer(model_name, len(LABEL_LIST))
        final_model.to(device)

        final_train_inputs, final_train_labels, _ = preprocess_data_for_token_classification(train_data, final_tokenizer, MAX_LEN)
        final_train_dataset = PatientNotesTokenClassificationDataset(final_train_inputs, final_train_labels)
        final_train_dataloader = DataLoader(final_train_dataset, batch_size=best_hyperparams['batch_size'], shuffle=True)

        final_val_inputs, final_val_labels, _ = preprocess_data_for_token_classification(val_data, final_tokenizer, MAX_LEN)
        final_val_dataset = PatientNotesTokenClassificationDataset(final_val_inputs, final_val_labels)
        final_val_dataloader = DataLoader(final_val_dataset, batch_size=best_hyperparams['batch_size'])

        final_optimizer = optim.AdamW(final_model.parameters(), lr=best_hyperparams['learning_rate'])

        for epoch in range(EPOCHS):
            train_and_validate_epoch(final_model, final_train_dataloader, final_val_dataloader, final_optimizer, device, epoch, EPOCHS)

        # Evaluate final model
        print(f"\n--- Final evaluation of {friendly_name} on the test set ---")
        test_jaccard = evaluate_model_on_data(final_model, final_tokenizer, test_data, device, MAX_LEN, LABEL_MAP)
        
        final_results[friendly_name] = {
            'best_params_from_cv': best_hyperparams,
            'avg_cv_jaccard': best_cv_score,
            'final_test_jaccard': test_jaccard
        }

    # Summary
    print(f"\n\n{'='*25} FINAL RESULTS {'='*25}")
    for model_name, result_data in final_results.items():
        print(f"Model: {model_name}")
        print(f"  - Best Params (from CV): {result_data['best_params_from_cv']}")
        print(f"  - Average Jaccard on CV Folds: {result_data['avg_cv_jaccard']:.4f}")
        print(f"  - Jaccard on Final Test Set: {result_data['final_test_jaccard']:.4f}")
        print("-" * 65)

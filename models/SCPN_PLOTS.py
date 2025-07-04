import os
import pandas as pd
import numpy as np
import torch
import ast
import itertools
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Config & Label definition
LABEL_LIST = ['O', 'B-FEATURE', 'I-FEATURE']
LABEL_MAP = {label: i for i, label in enumerate(LABEL_LIST)}
# Reverse map for easier lookup: ID to Label
ID_TO_LABEL = {i: label for label, i in LABEL_MAP.items()}
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
            # Only label tokens from the context (sequence_id == 1)
            # Question and special tokens (sequence_id 0 or None) are IGNORE_LABEL_ID
            if sequence_ids[i] == 1:
                token_labels[i] = LABEL_MAP['O'] # Initialize all context tokens to 'O'

        try:
            char_spans = ast.literal_eval(location_str)
        except (ValueError, SyntaxError):
            char_spans = []

        parsed_char_spans = []
        for span_str in char_spans:
            try:
                start_char, end_char = map(int, span_str.split())
                if start_char < end_char:
                    parsed_char_spans.append((start_char, end_char))
            except ValueError:
                continue

        for start_char, end_char in parsed_char_spans:
            first_token_in_current_span = True
            for token_idx in range(len(input_ids)):
                if sequence_ids[token_idx] != 1: # Only consider tokens from the context
                    continue
                token_char_start, token_char_end = offset_mapping[token_idx]
                
                # Skip special tokens or padding tokens where offsets are (0,0) for non-context parts
                if token_char_start == token_char_end == 0 and sequence_ids[token_idx] != 1:
                    continue
                
                # Check for overlap between character span and token span
                # An overlap exists if the max of the start points is less than the min of the end points
                if max(token_char_start, start_char) < min(token_char_end, end_char):
                    # Assign B-FEATURE to the first token in the span
                    if first_token_in_current_span:
                        token_labels[token_idx] = LABEL_MAP['B-FEATURE']
                        first_token_in_current_span = False
                    # Assign I-FEATURE to subsequent tokens in the span
                    else:
                        # Ensure we don't overwrite a B-FEATURE if there's an overlap from another span
                        # This part handles cases where spans might slightly overlap or touch
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
    print(f"Loading model and tokenizer for: {model_name}")
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Loaded tokenizer type: {type(tokenizer).__name__}")
    if "Fast" in type(tokenizer).__name__:
        print("Confirmed: A 'Fast' tokenizer was loaded, which supports offsets.")
    else:
        print("Warning: A non-'Fast' tokenizer was loaded. Offsets might not be supported or optimal.")
    return model, tokenizer

# Training and validation methods
def train_and_evaluate_epoch(model, train_dataloader, test_dataloader, optimizer, device, epoch_num, total_epochs):

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

    # Evaluation
    model.eval()
    total_test_loss = 0
    test_progress_bar = tqdm(test_dataloader, desc=f"Epoch {epoch_num+1}/{total_epochs} Evaluation", leave=False)
    with torch.no_grad():
        for batch in test_progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_test_loss += loss.item()
            test_progress_bar.set_postfix({'test_loss': loss.item()})

    avg_test_loss = total_test_loss / len(test_dataloader)
    print(f"Epoch {epoch_num+1} - Avg. Test Loss: {avg_test_loss:.4f}")

    return avg_train_loss, avg_test_loss

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
        is_context = sequence_ids and sequence_ids[i] == 1 # Check if token belongs to context
        if not is_context: # If not context, end any active span and continue
            if active_span_tokens:
                start, end = active_span_tokens[0][0], active_span_tokens[-1][1]
                if start < end: predicted_char_sets.append(set(range(start, end)))
                active_span_tokens = []
            continue

        token_char_start, token_char_end = offset_mapping[i]
        is_valid_token = not (token_char_start == 0 and token_char_end == 0) # Check for valid offset mapping

        if label_id == b_feature_id and is_valid_token:
            if active_span_tokens: # If a span was active, save it before starting a new one
                start, end = active_span_tokens[0][0], active_span_tokens[-1][1]
                if start < end: predicted_char_sets.append(set(range(start, end)))
            active_span_tokens = [(token_char_start, token_char_end)]
        elif label_id == i_feature_id and is_valid_token and active_span_tokens: # Continue current span
            active_span_tokens.append((token_char_start, token_char_end))
        else: # Label is 'O' or other invalid, end current span
            if active_span_tokens:
                start, end = active_span_tokens[0][0], active_span_tokens[-1][1]
                if start < end: predicted_char_sets.append(set(range(start, end)))
                active_span_tokens = []

    # After loop, if any span is still active, save it
    if active_span_tokens:
        start, end = active_span_tokens[0][0], active_span_tokens[-1][1]
        if start < end: predicted_char_sets.append(set(range(start, end)))

    return predicted_char_sets


def calculate_jaccard_and_collect_errors(model, tokenizer, df_to_eval, device, max_len, label_map, id_to_label, description="Evaluating Jaccard"):
    model.eval()
    eval_results = [] 

    _, true_labels_tensor, eval_info = preprocess_data_for_token_classification(df_to_eval, tokenizer, max_len)

    for i in tqdm(range(len(df_to_eval)), desc=description):
        row = eval_info['original_df'].iloc[i]
        question = str(row['feature_text'])
        context = str(row['pn_history'])
        true_location_str = str(row['location'])

        # Get the input_ids for this specific example
        input_ids_tensor = tokenizer.encode_plus(
            question, context,
            max_length=max_len, padding="max_length", truncation="only_second", return_tensors="pt"
        )['input_ids'].to(device)

        with torch.no_grad():
            outputs = model(input_ids_tensor)
            # Take argmax across the label dimension (dim=2) and squeeze to remove batch dim
            predicted_label_ids = torch.argmax(outputs.logits, dim=2).squeeze().tolist()


        # Get original tokens (useful for displaying errors)
        tokens = tokenizer.convert_ids_to_tokens(input_ids_tensor.squeeze().tolist())
        true_token_label_ids = true_labels_tensor[i].tolist()

        offset_mapping = eval_info['offset_mapping'][i]
        sequence_ids = eval_info['sequence_ids'][i]
        
        predicted_char_sets = extract_char_spans_from_token_labels(
            predicted_label_ids, offset_mapping, sequence_ids, label_map
        )
        true_char_sets = get_char_sets_from_location_str(true_location_str, len(eval_info['contexts'][i]))
        jaccard_score = calculate_overall_example_jaccard(predicted_char_sets, true_char_sets)

        eval_results.append({
            'jaccard_score': jaccard_score,
            'feature_text': question,
            'pn_history': context,
            'true_location_str': true_location_str,
            'predicted_label_ids': predicted_label_ids,
            'true_label_ids': true_token_label_ids,
            'tokens': tokens,
            'offset_mapping': offset_mapping,
            'sequence_ids': sequence_ids,
            'row_index': i 
        })

    # Sort results by Jaccard score (ascending, so worst scores are first)
    eval_results.sort(key=lambda x: x['jaccard_score'])

    mean_jaccard = np.mean([res['jaccard_score'] for res in eval_results])
    return mean_jaccard, eval_results

# Plots (no changes needed)
def save_performance_plot(history, model_friendly_name):
    """
    Speichert einen Plot von Trainings-/Test-Loss und Test-Jaccard-Score.
    """
    epochs_range = range(1, len(history['train_loss']) + 1)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss (lower is better)', color='blue')
    ax1.plot(epochs_range, history['train_loss'], 'o-', label='Trainings-Loss', color='blue')
    ax1.plot(epochs_range, history['test_loss'], 'o--', label='Test-Loss', color='cyan')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Jaccard Score (higher is better)', color='green')
    ax2.plot(epochs_range, history['test_jaccard'], 's-', label='Test-Jaccard Score', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.legend(loc='upper right')

    plt.title(f'Progress {model_friendly_name}')
    fig.tight_layout()

    safe_filename = model_friendly_name.replace('/', '_')
    try:
        base_dir = os.path.dirname(__file__)
    except NameError:
        base_dir = '.'

    output_dir = os.path.join(base_dir, 'nbme-score-clinical-patient-notes', 'plots')
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = os.path.join(output_dir, f'{safe_filename}_performance_plot.png')
    plt.savefig(plot_filename)
    print(f"\nPerformance plot saved to '{plot_filename}'")

# Helper to visualize token-level errors
def visualize_token_errors(example_data, tokenizer, id_to_label, top_n=10):
    print(f"\n{'='*20} TOP {top_n} WORST JACCARD SCORE EXAMPLES {'='*20}")
    print("Green: Correctly predicted feature token (B/I-FEATURE)")
    print("Red: Misclassified token (e.g., true B/I-FEATURE, predicted O, or vice versa)")
    print("Yellow: True O, Predicted O (Correctly ignored)")
    print("Grey: Special tokens or Question tokens (ignored for scoring)")
    print("-" * 70)

    for i, ex in enumerate(example_data[:top_n]):
        feature_text = ex['feature_text']
        pn_history = ex['pn_history']
        tokens = ex['tokens']
        true_labels = ex['true_label_ids']
        predicted_labels = ex['predicted_label_ids']
        sequence_ids = ex['sequence_ids']
        jaccard_score = ex['jaccard_score']
        offset_mapping = ex['offset_mapping']

        print(f"\n--- Example {i+1} (Jaccard: {jaccard_score:.4f}) ---")
        print(f"Feature: {feature_text}")
        print(f"Patient History (Context): {pn_history}")
        print("\nToken-level breakdown:")

        # Reconstruct text with highlighted tokens for better readability
        display_text_parts = []
        for token_idx in range(len(tokens)):
            token = tokens[token_idx]
            true_label_id = true_labels[token_idx]
            predicted_label_id = predicted_labels[token_idx]
            seq_id = sequence_ids[token_idx]
            
            # Get original token text from context using offset mapping
            token_char_start, token_char_end = offset_mapping[token_idx]
            original_token_text = pn_history[token_char_start:token_char_end] if seq_id == 1 and token_char_start < token_char_end else token
            
            # Determine color/style based on classification
            if seq_id == 1: # Only evaluate context tokens
                if true_label_id == predicted_label_id:
                    # Correctly predicted 'O' (non-feature)
                    if id_to_label[true_label_id] == 'O':
                        color_code = '\033[93m' # Yellow for correct 'O'
                    # Correctly predicted B/I-FEATURE
                    else:
                        color_code = '\033[92m' # Green for correct feature
                else:
                    # Misclassified token
                    color_code = '\033[91m' # Red for error
                reset_code = '\033[0m'

                # Add labels for clarity
                display_text_parts.append(
                    f"{color_code}{original_token_text} "
                    f"[{id_to_label.get(true_label_id, 'N/A')}/{id_to_label.get(predicted_label_id, 'N/A')}]{reset_code}"
                )
            else: # Special tokens or question tokens
                display_text_parts.append(f"\033[90m{token}\033[0m") # Grey

        print(' '.join(display_text_parts))
        print("-" * 70)


# Main execution block
if __name__ == '__main__':

    # Test mode with a small subset
    RUN_IN_LOCAL_MODE = False 


    # DATA_PATH = "nbme-score-clinical-patient-notes/" # for HKA GPU
    DATA_PATH = "ClinicalData/nbme-score-clinical-patient-notes/" # local 
    MAX_LEN = 350
    EPOCHS = 3 # Reduced epochs for faster testing

    MODELS_TO_TRAIN = [
        {
            'friendly_name': 'BioFormer-16L',
            'model_path': 'bioformers/BioFormer-16L',
            'learning_rate': 5e-5,
            'batch_size': 16
        }
    ]

    # Setup
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
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

    if RUN_IN_LOCAL_MODE:
        print("\n--- RUNNING IN LOCAL MODE (sampling 100 entries) ---")
        df_to_use = merged_df.sample(n=100, random_state=42).copy()
    else:
        print("\n--- RUNNING IN FULL-DATA GPU MODE ---")
        df_to_use = merged_df.copy()

    print(f"Size of dataset being used: {len(df_to_use)} entries")

    final_results = {}

    # Loop for evaluation
    for model_config in MODELS_TO_TRAIN:
        friendly_name = model_config['friendly_name']
        model_path = model_config['model_path']
        lr = model_config['learning_rate']
        batch_size = model_config['batch_size']

        print(f"\n\n{'='*20} TRAINING AND EVALUATING: {friendly_name.upper()} {'='*20}")
        print(f"Params: LR={lr}, Batch Size={batch_size}")

        # Data split for final training and testing
        train_data, test_data = train_test_split(df_to_use, test_size=0.2, random_state=42)

        print(f"Training data size: {len(train_data)}")
        print(f"Test data size: {len(test_data)}")

        current_model, current_tokenizer = create_model_and_tokenizer(model_path, len(LABEL_LIST))
        current_model.to(device)

        print("Preprocessing training data...")
        train_inputs, train_labels, _ = preprocess_data_for_token_classification(train_data, current_tokenizer, MAX_LEN)
        train_dataset = PatientNotesTokenClassificationDataset(train_inputs, train_labels)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        print("Preprocessing test data...")
        test_inputs, test_labels, _ = preprocess_data_for_token_classification(test_data, current_tokenizer, MAX_LEN)
        test_dataset = PatientNotesTokenClassificationDataset(test_inputs, test_labels)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)


        optimizer = optim.AdamW(current_model.parameters(), lr=lr)

        # Train and evaluate model
        print(f"\n--- Training {friendly_name} for {EPOCHS} epochs ---")

        training_history = {'train_loss': [], 'test_loss': [], 'test_jaccard': []}
        all_eval_results_for_model = [] 

        for epoch in range(EPOCHS):
            avg_train_loss, avg_test_loss = train_and_evaluate_epoch(current_model, train_dataloader, test_dataloader, optimizer, device, epoch, EPOCHS)
            training_history['train_loss'].append(avg_train_loss)
            training_history['test_loss'].append(avg_test_loss)

            
            jaccard_score_epoch, current_eval_results = calculate_jaccard_and_collect_errors(
                current_model, current_tokenizer, test_data, device, MAX_LEN, LABEL_MAP, ID_TO_LABEL,
                description=f"Epoch {epoch+1}/{EPOCHS} Jaccard Eval"
            )
            print(f"Epoch {epoch+1} - Test Jaccard: {jaccard_score_epoch:.4f}")
            training_history['test_jaccard'].append(jaccard_score_epoch)

            if epoch == EPOCHS - 1:
                all_eval_results_for_model = current_eval_results

        save_performance_plot(training_history, friendly_name)

        best_jaccard_score = max(training_history['test_jaccard'])
        final_jaccard_score = training_history['test_jaccard'][-1]

        print(f"\n--- Final results for {friendly_name} ---")
        print(f"Jaccard score in final epoch: {final_jaccard_score:.4f}")
        print(f"Best Jaccard score during training: {best_jaccard_score:.4f}")

        final_results[friendly_name] = {
            'learning_rate_used': lr,
            'batch_size_used': batch_size,
            'final_test_jaccard': final_jaccard_score,
            'best_test_jaccard_during_training': best_jaccard_score
        }

        # Display worst performing examples and token-level errors 
        if all_eval_results_for_model:
            visualize_token_errors(all_eval_results_for_model, current_tokenizer, ID_TO_LABEL, top_n=10)


    # Summary
    print(f"\n\n{'='*25} FINAL RESULTS SUMMARY {'='*25}")
    for model_name, result_data in final_results.items():
        print(f"Model: {model_name}")
        print(f"  - Learning Rate: {result_data['learning_rate_used']}")
        print(f"  - Batch Size: {result_data['batch_size_used']}")
        print(f"  - Best Test Jaccard during Training: {result_data['best_test_jaccard_during_training']:.4f}")
        print(f"  - Test Jaccard in Final Epoch: {result_data['final_test_jaccard']:.4f}")
        print("-" * 65)
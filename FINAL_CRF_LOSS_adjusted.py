import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import ast
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torchcrf import CRF
import statsmodels.api as sm
import time

LABEL_LIST = ['O', 'B-FEATURE', 'I-FEATURE']
LABEL_MAP = {label: i for i, label in enumerate(LABEL_LIST)}
INV_LABEL_MAP = {i: label for label, i in LABEL_MAP.items()}
IGNORE_LABEL_ID = -100 # Special ID for tokens to ignore in loss

class JaccardLoss(nn.Module):
    # JaccardLoss custom loss function for token classification
    def __init__(self, smooth=1e-6, ignore_index=IGNORE_LABEL_ID):
        super(JaccardLoss, self).__init__()
        self.smooth = smooth # Smoothing factor to prevent division by zero
        self.ignore_index = ignore_index # Label ID to ignore in loss calculation

    def forward(self, logits, labels):
        probs = F.softmax(logits, dim=-1) # Convert logits to probabilities
        num_classes = logits.shape[-1]
        
        # Reshape for loss calculation
        flat_probs = probs.view(-1, num_classes)
        flat_labels = labels.view(-1)
        
        # Create one-hot encoded labels, ignoring specified index
        active_mask = flat_labels != self.ignore_index # Mask to exclude ignored tokens
        active_labels = flat_labels[active_mask]
        active_probs = flat_probs[active_mask]
        
        if active_labels.numel() == 0:
            return torch.tensor(0.0, device=logits.device) # Return 0 loss if no active labels
            
        true_one_hot = F.one_hot(active_labels, num_classes=num_classes).float()
        
        # Ignore the 'O' class (index 0) from Jaccard calculation
        target_probs = active_probs[:, 1:] # Exclude 'O' class probabilities
        target_true = true_one_hot[:, 1:] # Exclude 'O' class from true labels
        
        intersection = torch.sum(target_probs * target_true, dim=0)
        union = torch.sum(target_probs, dim=0) + torch.sum(target_true, dim=0) - intersection
        
        jaccard_score = (intersection + self.smooth) / (union + self.smooth)
        jaccard_loss = 1 - torch.mean(jaccard_score) # Convert score to loss
        
        return jaccard_loss

class BertCRF(nn.Module):
    # BertCRF combines a BERT model with a CRF layer for sequence labeling
    def __init__(self, model_name, num_labels):
        super(BertCRF, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name, ignore_mismatched_sizes=True) # Load pre-trained BERT model
        self.dropout = nn.Dropout(0.1) # Dropout layer for regularization
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels) # Linear layer for classification
        self.crf = CRF(num_labels, batch_first=True) # Conditional Random Field layer
        self.jaccard_loss_fn = JaccardLoss(ignore_index=IGNORE_LABEL_ID) # Jaccard loss instance

    def forward(self, input_ids, attention_mask, labels=None, alpha=1.0):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0] # Hidden states from BERT
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output) # Logits before CRF

        if labels is not None:
            active_labels = labels.clone()
            # The CRF layer cannot handle ignore_index, so we map it to a valid label (e.g., 'O')
            # The attention mask will ensure these positions do not contribute to the loss.
            active_labels[labels == IGNORE_LABEL_ID] = LABEL_MAP['O'] # Replace ignore_index for CRF
            
            mask = attention_mask.bool() # Convert attention mask to boolean
            # Calculate CRF loss
            crf_loss = -self.crf(logits, active_labels, mask=mask, reduction='mean') # Negative log likelihood from CRF
            jaccard_loss = self.jaccard_loss_fn(logits, labels) # Jaccard loss can handle ignore_index

            # Combine losses
            loss = alpha * crf_loss + (1 - alpha) * jaccard_loss # Weighted sum of losses
            return loss, logits
        else:
            predictions = self.crf.decode(logits, mask=attention_mask.bool()) # Decode using CRF for inference
            return predictions, logits

def preprocess_data_for_token_classification(df, tokenizer, max_len):
    # Preprocesses dataframe for token classification, tokenizing and aligning labels
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
            truncation="only_second", # Truncate only the second sequence (context)
            return_offsets_mapping=True, # Important for mapping tokens back to characters
            return_attention_mask=True
        )

        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        offset_mapping = encoding['offset_mapping']
        sequence_ids = encoding.sequence_ids() # Identify which token belongs to which sequence

        token_labels = [IGNORE_LABEL_ID] * len(input_ids) # Initialize labels with ignore ID

        for i in range(len(input_ids)):
            if sequence_ids[i] == 1: # Only label tokens from the context
                token_labels[i] = LABEL_MAP['O'] # Default to 'O' (Outside)

        try:
            char_spans = ast.literal_eval(location_str) # Parse location string to list of spans
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
                if sequence_ids[token_idx] != 1:
                    continue # Skip tokens not in the context
                token_char_start, token_char_end = offset_mapping[token_idx]
                if token_char_start == token_char_end == 0:
                    continue # Skip special tokens or padding
                # Check for overlap between token span and true character span
                if max(token_char_start, start_char) < min(token_char_end, end_char):
                    if first_token_in_current_span:
                        token_labels[token_idx] = LABEL_MAP['B-FEATURE'] # Mark beginning of a feature
                        first_token_in_current_span = False
                    else:
                        if token_labels[token_idx] != LABEL_MAP['B-FEATURE']: # Avoid overwriting 'B'
                            token_labels[token_idx] = LABEL_MAP['I-FEATURE'] # Mark inside a feature

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


class PatientNotesTokenClassificationDataset(Dataset):
    # Custom Dataset for patient notes token classification
    def __init__(self, encodings_list, labels_list):
        self.encodings_list = encodings_list
        self.labels_list = labels_list

    def __getitem__(self, idx):
        item = self.encodings_list[idx]
        item['labels'] = self.labels_list[idx] # Add labels to the item
        return item

    def __len__(self):
        return len(self.labels_list)

def create_model_and_tokenizer(model_name: str, num_labels: int):
    # Creates an instance of BertCRF model and its corresponding tokenizer
    print(f"Loading model and tokenizer for: {model_name}")
    model = BertCRF(model_name, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Loaded tokenizer type: {type(tokenizer).__name__}")
    if "Fast" in type(tokenizer).__name__:
        print("Confirmed: A 'Fast' tokenizer was loaded, which supports offsets.")
    else:
        print("Warning: A non-'Fast' tokenizer was loaded. Offsets might not be supported or optimal.")
    return model, tokenizer

def train_and_evaluate_epoch_hybrid(model, train_dataloader, test_dataloader, optimizer, device, epoch_num, total_epochs, alpha):
    # Trains and evaluates the model for one epoch using a hybrid loss
    model.train() # Set model to training mode
    total_train_loss = 0
    train_progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch_num+1}/{total_epochs} Training (alpha={alpha:.2f})", leave=False)

    for batch in train_progress_bar:
        optimizer.zero_grad() # Clear previous gradients
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        loss, _ = model(input_ids, attention_mask=attention_mask, labels=labels, alpha=alpha) # Forward pass
        total_train_loss += loss.item()
        loss.backward() # Backpropagate
        optimizer.step() # Update model parameters
        train_progress_bar.set_postfix({'loss': loss.item()})

    avg_train_loss = total_train_loss / len(train_dataloader)

    model.eval() # Set model to evaluation mode
    total_test_loss = 0
    test_progress_bar = tqdm(test_dataloader, desc=f"Epoch {epoch_num+1}/{total_epochs} Evaluation", leave=False)
    with torch.no_grad(): # Disable gradient calculation for evaluation
        for batch in test_progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            loss, _ = model(input_ids, attention_mask=attention_mask, labels=labels, alpha=alpha)
            total_test_loss += loss.item()

    avg_test_loss = total_test_loss / len(test_dataloader)
   
    return avg_train_loss, avg_test_loss

def grid_search_loss_weights(model_config, full_train_df, device, max_len):
    # Performs a grid search to find the optimal alpha weight for the hybrid loss
    print("\n--- Starting Grid Search for Optimal Loss Weights ---")
    
    search_df = full_train_df.sample(frac=0.1, random_state=42) # Use a small subset for faster search
    search_train_df, search_val_df = train_test_split(search_df, test_size=0.2, random_state=42)

    model_path = model_config['model_path']
    lr = model_config['learning_rate']
    batch_size = model_config['batch_size']

    model, tokenizer = create_model_and_tokenizer(model_path, len(LABEL_LIST))
    model.to(device)

    train_inputs, train_labels, _ = preprocess_data_for_token_classification(search_train_df, tokenizer, max_len)
    train_dataset = PatientNotesTokenClassificationDataset(train_inputs, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_inputs, val_labels, val_eval_info = preprocess_data_for_token_classification(search_val_df, tokenizer, max_len)
    val_dataset = PatientNotesTokenClassificationDataset(val_inputs, val_labels)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    #alpha_grid = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0] # Candidate alpha values
    alpha_grid = [0.1, 0.5, 0.7, 1.0] # Reduced grid for demonstration
    best_alpha = 1.0
    best_jaccard = -1.0

    for alpha in alpha_grid:
        print(f"\nGrid Search: Testing alpha = {alpha}")
        # Reset model and optimizer for each alpha value to ensure fair comparison
        model, _ = create_model_and_tokenizer(model_path, len(LABEL_LIST))
        model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        
        for epoch in range(3): # Train for a few epochs for each alpha
            train_and_evaluate_epoch_hybrid(model, train_dataloader, val_dataloader, optimizer, device, epoch, 3, alpha)
        
        val_jaccard, _ = evaluate_metrics(model, val_dataloader, device, val_eval_info, LABEL_MAP, description=f"Grid Search Eval (alpha={alpha})")
        print(f"Alpha {alpha} -> Validation Jaccard: {val_jaccard:.4f}")

        if val_jaccard > best_jaccard:
            best_jaccard = val_jaccard
            best_alpha = alpha

    print(f"\n--- Grid Search Complete ---")
    print(f"Best Alpha Found: {best_alpha} (Jaccard: {best_jaccard:.4f})")
    return best_alpha

def get_char_sets_from_location_str(location_str: str, pn_text_len: int) -> list[set[int]]:
    # Converts a location string into a list of character sets (spans)
    true_spans_char_sets = []
    try:
        char_spans_from_str = ast.literal_eval(location_str)
        for span_str in char_spans_from_str:
            parts = span_str.split()
            if len(parts) == 2:
                start_char, end_char = int(parts[0]), int(parts[1])
                start_char = max(0, start_char) # Ensure start is not negative
                end_char = min(pn_text_len, end_char) # Ensure end is within text bounds
                if start_char < end_char:
                    true_spans_char_sets.append(set(range(start_char, end_char))) # Create set of character indices
    except (ValueError, SyntaxError):
        pass
    return true_spans_char_sets

def calculate_overall_example_jaccard(predicted_char_sets: list[set[int]], true_char_sets: list[set[int]]) -> float:
    # Calculates Jaccard score for a single example based on predicted and true character sets
    union_predicted_chars = set().union(*predicted_char_sets) # Union of all predicted character sets
    union_true_chars = set().union(*true_char_sets) # Union of all true character sets
    if not union_predicted_chars and not union_true_chars: return 1.0 # Perfect match if both are empty
    if not union_predicted_chars or not union_true_chars: return 0.0 # No overlap if one is empty
    intersection_of_unions = len(union_predicted_chars.intersection(union_true_chars))
    union_of_unions = len(union_predicted_chars.union(union_true_chars))
    return intersection_of_unions / union_of_unions if union_of_unions > 0 else 1.0

def extract_char_spans_from_token_labels(token_label_ids, offset_mapping, sequence_ids, label_map):
    # Extracts character spans from predicted token labels
    predicted_char_sets = []
    active_span_tokens = []
    b_feature_id = label_map.get('B-FEATURE')
    i_feature_id = label_map.get('I-FEATURE')

    for i, label_id in enumerate(token_label_ids):
        is_context = sequence_ids and sequence_ids[i] == 1 # Check if token belongs to context
        if not is_context:
            if active_span_tokens: # Close any active span if outside context
                start, end = active_span_tokens[0][0], active_span_tokens[-1][1]
                if start < end: predicted_char_sets.append(set(range(start, end)))
                active_span_tokens = []
            continue

        token_char_start, token_char_end = offset_mapping[i]
        is_valid_token = not (token_char_start == 0 and token_char_end == 0) # Exclude padding/special tokens

        if label_id == b_feature_id and is_valid_token:
            if active_span_tokens: # If a span was active, finalize it
                start, end = active_span_tokens[0][0], active_span_tokens[-1][1]
                if start < end: predicted_char_sets.append(set(range(start, end)))
            active_span_tokens = [(token_char_start, token_char_end)] # Start new span
        elif label_id == i_feature_id and is_valid_token and active_span_tokens:
            active_span_tokens.append((token_char_start, token_char_end)) # Continue current span
        else:
            if active_span_tokens: # If label is not B or I, close active span
                start, end = active_span_tokens[0][0], active_span_tokens[-1][1]
                if start < end: predicted_char_sets.append(set(range(start, end)))
                active_span_tokens = []

    if active_span_tokens: # Close any remaining active span at the end
        start, end = active_span_tokens[0][0], active_span_tokens[-1][1]
        if start < end: predicted_char_sets.append(set(range(start, end)))

    return predicted_char_sets

def evaluate_metrics(model, dataloader, device, eval_info, label_map, description="Evaluating"):
    # Evaluates the model's performance using Jaccard and Micro F1 scores
    model.eval()
    all_jaccards = []
    all_true_labels_for_f1 = []
    all_pred_labels_for_f1 = []

    example_idx = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=description, leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
           
            predictions, _ = model(input_ids, attention_mask=attention_mask) # Get model predictions

            for i in range(input_ids.shape[0]): # Iterate through each example in the batch
                pred_labels_i = predictions[i]
                true_labels_i = labels[i].tolist()
               
                active_mask = [lbl != IGNORE_LABEL_ID for lbl in true_labels_i] # Filter out ignored tokens
                active_true = [lbl for j, lbl in enumerate(true_labels_i) if active_mask[j]]
                active_pred = [lbl for j, lbl in enumerate(pred_labels_i) if active_mask[j]]
               
                all_true_labels_for_f1.extend(active_true) # Collect for F1 calculation
                all_pred_labels_for_f1.extend(active_pred)

                offset_map = eval_info['offset_mapping'][example_idx]
                seq_ids = eval_info['sequence_ids'][example_idx]
                context = eval_info['contexts'][example_idx]
                true_loc_str = eval_info['original_df'].iloc[example_idx]['location']

                predicted_char_sets = extract_char_spans_from_token_labels(
                    pred_labels_i, offset_map, seq_ids, label_map
                )
                true_char_sets = get_char_sets_from_location_str(true_loc_str, len(context))
                jaccard_score = calculate_overall_example_jaccard(predicted_char_sets, true_char_sets)
                all_jaccards.append(jaccard_score)

                example_idx += 1

    mean_jaccard = np.mean(all_jaccards) if all_jaccards else 0.0
    micro_f1 = f1_score(all_true_labels_for_f1, all_pred_labels_for_f1, average='micro', zero_division=0)
   
    return mean_jaccard, micro_f1

def perform_final_detailed_evaluation(model, dataloader, device, eval_info, label_map):
    # Performs a detailed evaluation, collecting data for various analysis plots
    model.eval()
    all_jaccards = []
    all_note_lengths = []
    all_true_labels = []
    all_pred_labels = []
   
    example_idx = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Final Detailed Evaluation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
           
            predictions, _ = model(input_ids, attention_mask=attention_mask)

            for i in range(input_ids.shape[0]):
                pred_labels_i = predictions[i]
                true_labels_i = labels[i].tolist()
               
                active_mask = [lbl != IGNORE_LABEL_ID for lbl in true_labels_i]
                active_true = [lbl for j, lbl in enumerate(true_labels_i) if active_mask[j]]
                active_pred = [lbl for j, lbl in enumerate(pred_labels_i) if active_mask[j]]
               
                all_true_labels.extend(active_true)
                all_pred_labels.extend(active_pred)

                offset_map = eval_info['offset_mapping'][example_idx]
                seq_ids = eval_info['sequence_ids'][example_idx]
                context = eval_info['contexts'][example_idx]
                true_loc_str = eval_info['original_df'].iloc[example_idx]['location']

                predicted_char_sets = extract_char_spans_from_token_labels(
                    pred_labels_i, offset_map, seq_ids, label_map
                )
                true_char_sets = get_char_sets_from_location_str(true_loc_str, len(context))
                jaccard_score = calculate_overall_example_jaccard(predicted_char_sets, true_char_sets)
                all_jaccards.append(jaccard_score)
                all_note_lengths.append(len(context)) # Store context length
               
                example_idx += 1
               
    results = {
        "jaccard_scores_per_example": all_jaccards,
        "note_lengths": all_note_lengths,
        "true_token_labels": all_true_labels,
        "pred_token_labels": all_pred_labels
    }
    return results

def get_output_dir():
    # Defines and creates the output directory for plots and metrics
    try:
        base_dir = os.path.dirname(__file__)
    except NameError:
        base_dir = '.'
    output_dir = os.path.join(base_dir, 'nbme-score-clinical-patient-notes', 'plots_and_metrics_LOSS_adjusted')
    os.makedirs(output_dir, exist_ok=True) # Create directory if it doesn't exist
    return output_dir

def plot_combined_training_curves(all_histories):
    # Plots combined training and test loss/metric curves for all models
    output_dir = get_output_dir()
   
    fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    fig.suptitle('Model Convergence and Stability Comparison', fontsize=16)

    metrics = [('Loss', 'loss'), ('Jaccard Score', 'jaccard'), ('Micro F1-Score', 'f1')]
    colors = plt.cm.jet(np.linspace(0, 1, len(all_histories) * 2)) # Generate distinct colors
   
    for ax, (metric_name, metric_key) in zip(axes, metrics):
        color_idx = 0
        for model_name, history in all_histories.items():
            epochs = range(1, len(history['train_loss']) + 1)
            ax.plot(epochs, history[f'train_{metric_key}'], 'o-', label=f'{model_name} Train', color=colors[color_idx])
            ax.plot(epochs, history[f'test_{metric_key}'], 's--', label=f'{model_name} Test', color=colors[color_idx+1])
            color_idx += 2
        ax.set_ylabel(metric_name)
        ax.grid(True)
        ax.legend()

    axes[-1].set_xlabel('Epochs')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    filepath = os.path.join(output_dir, '1_combined_training_curves.png')
    plt.savefig(filepath)
    plt.close()
    print(f"Combined training curves saved to '{filepath}'")

def plot_performance_gap(final_results_summary):
    # Plots the best Jaccard and Micro F1 scores for each model as a bar chart
    output_dir = get_output_dir()
    model_names = list(final_results_summary.keys())
    jaccard_scores = [res['best_test_jaccard_during_training'] for res in final_results_summary.values()]
    f1_scores = [res['best_test_f1_during_training'] for res in final_results_summary.values()]

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, jaccard_scores, width, label='Jaccard Score')
    rects2 = ax.bar(x + width/2, f1_scores, width, label='Micro F1-Score')

    ax.set_ylabel('Scores')
    ax.set_title('Performance Gap Between Models (Best Epoch)')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.legend()
    ax.bar_label(rects1, padding=3, fmt='%.4f') # Add labels to bars
    ax.bar_label(rects2, padding=3, fmt='%.4f')
    ax.set_ylim(0, 1)

    plt.tight_layout()
    filepath = os.path.join(output_dir, '2_performance_gap.png')
    plt.savefig(filepath)
    plt.close()
    print(f"Performance gap plot saved to '{filepath}'")

def plot_per_label_metrics(all_detailed_results):
    # Plots precision, recall, and F1-score for each label (excluding 'O')
    output_dir = get_output_dir()
   
    for model_name, results in all_detailed_results.items():
        true_labels = results['true_token_labels']
        pred_labels = results['pred_token_labels']
       
        filtered_true, filtered_pred = [], []
        for t, p in zip(true_labels, pred_labels):
            if t != LABEL_MAP['O'] or p != LABEL_MAP['O']: # Only include non-'O' labels or predictions of non-'O'
                filtered_true.append(t)
                filtered_pred.append(p)

        labels_to_include = [l for l in LABEL_LIST if l != 'O'] # Focus on 'B-FEATURE' and 'I-FEATURE'
        label_ids_to_include = [LABEL_MAP[l] for l in labels_to_include]

        p, r, f1, _ = precision_recall_fscore_support(filtered_true, filtered_pred, labels=label_ids_to_include, average=None, zero_division=0)
       
        metrics_df = pd.DataFrame({'Precision': p, 'Recall': r, 'F1-Score': f1}, index=labels_to_include)
       
        ax = metrics_df.plot(kind='bar', figsize=(12, 7), rot=0)
        plt.title(f'Per-Label Performance: {model_name}')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', label_type='edge')
       
        plt.tight_layout()
        safe_filename = model_name.replace('/', '_') # Sanitize filename
        filepath = os.path.join(output_dir, f'3_per_label_metrics_{safe_filename}.png')
        plt.savefig(filepath)
        plt.close()
        print(f"Per-label metrics plot for {model_name} saved to '{filepath}'")

def plot_jaccard_distribution(all_detailed_results):
    # Plots the distribution of example-level Jaccard scores using violin plots
    output_dir = get_output_dir()
    
    plot_data = []
    for model_name, results in all_detailed_results.items():
        for score in results['jaccard_scores_per_example']:
            plot_data.append({'model_name': model_name, 'jaccard_score': score})
    
    df = pd.DataFrame(plot_data)
    
    plt.figure(figsize=(12, 7))
    # The fix is adding cut=0 here to limit violin plot extent
    sns.violinplot(data=df, x='model_name', y='jaccard_score', inner='quartile', cut=0)
    
    plt.xticks(rotation=45, ha="right")
    plt.title('Distribution of Example-Level Jaccard Scores')
    plt.ylabel('Jaccard Score')
    plt.xlabel('Model')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, '4_jaccard_distribution.png')
    plt.savefig(filepath)
    plt.close()
    print(f"Jaccard distribution plot saved to '{filepath}'")

def plot_note_length_vs_jaccard(all_detailed_results):
    # Plots Jaccard scores against note lengths with a LOWESS trend line
    output_dir = get_output_dir()
    num_models = len(all_detailed_results)
    fig, axes = plt.subplots(num_models, 1, figsize=(10, 6 * num_models), sharex=True, sharey=True)
    if num_models == 1: axes = [axes] # Handle single subplot case
    fig.suptitle('Note Length vs. Jaccard Score', fontsize=16)

    for ax, (model_name, results) in zip(axes, all_detailed_results.items()):
        note_lengths = results['note_lengths']
        jaccards = results['jaccard_scores_per_example']
       
        lowess = sm.nonparametric.lowess # Locally Weighted Scatterplot Smoothing
        z = lowess(jaccards, note_lengths, frac=0.3) # Calculate LOWESS trend

        ax.scatter(note_lengths, jaccards, alpha=0.2, label='Example')
        ax.plot(z[:, 0], z[:, 1], color='red', lw=3, label='LOWESS Trend')
        ax.set_title(model_name)
        ax.set_ylabel('Jaccard Score')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

    axes[-1].set_xlabel('Note Length (characters)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    filepath = os.path.join(output_dir, '5_note_length_vs_jaccard.png')
    plt.savefig(filepath)
    plt.close()
    print(f"Note length vs Jaccard plot saved to '{filepath}'")

def plot_confusion_matrix(all_detailed_results):
    # Plots normalized confusion matrices, one for all non-'O' labels and one specifically for 'B-FEATURE'
    output_dir = get_output_dir()
   
    for model_name, results in all_detailed_results.items():
        true_labels_int = results['true_token_labels']
        pred_labels_int = results['pred_token_labels']
       
        labels_to_include = [l for l in LABEL_LIST if l != 'O'] # Exclude 'O' for clearer visualization of features
        label_ids_to_include = [LABEL_MAP[l] for l in labels_to_include]
       
        filtered_true, filtered_pred = [], []
        for t, p in zip(true_labels_int, pred_labels_int):
            if t in label_ids_to_include or p in label_ids_to_include: # Include if true or predicted label is a feature
                filtered_true.append(t)
                filtered_pred.append(p)

        if not filtered_true:
            print(f"Skipping confusion matrix for {model_name} - no non-'O' labels found.")
            continue

        cm = confusion_matrix(filtered_true, filtered_pred, labels=label_ids_to_include, normalize='true') # Normalized confusion matrix
       
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels_to_include, yticklabels=labels_to_include)
        plt.title(f'Normalized Confusion Matrix (O-Label Removed)\n{model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        safe_filename = model_name.replace('/', '_')
        filepath = os.path.join(output_dir, f'6a_confusion_matrix_{safe_filename}.png')
        plt.savefig(filepath)
        plt.close()
        print(f"Confusion matrix plot for {model_name} saved to '{filepath}'")

        b_label_id = LABEL_MAP['B-FEATURE']
        b_true = [t for t in true_labels_int if t != IGNORE_LABEL_ID] # Filter out ignored tokens
        b_pred = [p for t,p in zip(true_labels_int, pred_labels_int) if t != IGNORE_LABEL_ID]
       
        b_filtered_true, b_filtered_pred = [], []
        for t, p in zip(b_true, b_pred):
            is_true_b = (t == b_label_id)
            is_pred_b = (p == b_label_id)
            if is_true_b or is_pred_b: # Focus on 'B-FEATURE' and 'Not B-FEATURE'
                b_filtered_true.append('B-FEATURE' if is_true_b else 'Not B-FEATURE')
                b_filtered_pred.append('B-FEATURE' if is_pred_b else 'Not B-FEATURE')
       
        if b_filtered_true:
            b_labels = ['B-FEATURE', 'Not B-FEATURE']
            cm_b = confusion_matrix(b_filtered_true, b_filtered_pred, labels=b_labels, normalize='true')
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm_b, annot=True, fmt='.2f', cmap='Greens', xticklabels=b_labels, yticklabels=b_labels)
            plt.title(f'B-Token Confusion Matrix (Missed Starts)\n{model_name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            filepath_b = os.path.join(output_dir, f'6b_b_token_confusion_matrix_{safe_filename}.png')
            plt.savefig(filepath_b)
            plt.close()
            print(f"B-Token confusion matrix for {model_name} saved to '{filepath_b}'")

if __name__ == '__main__':
    RUN_IN_LOCAL_MODE = False # Set to True for quick local testing
    DATA_PATH = "nbme-score-clinical-patient-notes/"
    MAX_LEN = 300
    EPOCHS = 7

    MODELS_TO_TRAIN = [
        {
            'friendly_name': 'Roberta',
            'model_path': 'FacebookAI/roberta-large',
            'learning_rate': 1e-5,
            'batch_size': 6
        },
        {
            'friendly_name': 'Clinical-Bert',
            'model_path': 'AKHIL001/Bio_Clinical_BERT',
            'learning_rate': 7e-5,
            'batch_size': 8
        },
        {
            'friendly_name': 'BioFormer',
            'model_path': 'bioformers/BioFormer-16L',
            'learning_rate': 5e-5,
            'batch_size': 16
        },
        {
            'friendly_name': 'Bert-base',
            'model_path': 'bert-base-uncased',
            'learning_rate': 5e-5,
            'batch_size': 8
        },
        {
            'friendly_name': 'DeBerta',
            'model_path': 'MoritzLaurer/deberta-v3-large-zeroshot-v2.0',
            'learning_rate': 1e-5,
            'batch_size': 1
        },
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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
        print("\n--- RUNNING IN LOCAL MODE ---")
        df_to_use = merged_df.sample(n=1000, random_state=42).copy() # Use a smaller sample for local run
    else:
        print("\n--- RUNNING IN FULL-DATA GPU MODE ---")
        df_to_use = merged_df.copy()

    print(f"Size of dataset being used: {len(df_to_use)} entries")

    all_models_training_histories = {}
    all_models_detailed_results = {}
    final_results_summary = {}

    for model_config in MODELS_TO_TRAIN:
        friendly_name = model_config['friendly_name']
        model_path = model_config['model_path']
        lr = model_config['learning_rate']
        batch_size = model_config['batch_size']

        print(f"\n\n{'='*20} PROCESSING: {friendly_name.upper()} {'='*20}")
        print(f"Params: LR={lr}, Batch Size={batch_size}")

        train_data, test_data = train_test_split(df_to_use, test_size=0.2, random_state=42)
        
        # Grid search for the best alpha on 10% of the training data
        optimal_alpha = grid_search_loss_weights(model_config, train_data, device, MAX_LEN)

        print(f"\n--- Starting Full Training for {friendly_name} with optimal alpha={optimal_alpha:.2f} ---")
        print(f"Training data size: {len(train_data)}")
        print(f"Test data size: {len(test_data)}")

        current_model, current_tokenizer = create_model_and_tokenizer(model_path, len(LABEL_LIST))
        current_model.to(device)

        print("Preprocessing full training data...")
        train_inputs, train_labels, train_eval_info = preprocess_data_for_token_classification(train_data, current_tokenizer, MAX_LEN)
        train_dataset = PatientNotesTokenClassificationDataset(train_inputs, train_labels)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        train_eval_dataloader = DataLoader(train_dataset, batch_size=batch_size) # For evaluation on training set

        print("Preprocessing test data...")
        test_inputs, test_labels, test_eval_info = preprocess_data_for_token_classification(test_data, current_tokenizer, MAX_LEN)
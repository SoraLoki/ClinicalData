import os
import pandas as pd
import numpy as np
import torch
import ast
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# =====================================================================================
#  Data Preprocessing
# =====================================================================================
def preprocess_data_for_qa(df, tokenizer, max_len):
    # Prepares the data in a question-answering format.
    # Each row is treated as a single question-answer pair.
    all_input_ids = []
    all_attention_masks = []
    all_start_positions = []
    all_end_positions = []
    
    # Stores additional information required for evaluation.
    eval_info = {
        'offset_mapping': [],
        'sequence_ids': [],
        'contexts': [],
        'answer_texts': [],
        'original_df': df.copy() # Stores original dataframe rows for easy access to question/locations.
    }

    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Preprocessing data"):
        question = str(row['feature_text'])
        context = str(row['pn_history'])
        location_str = str(row['location'])
        
        encoding = tokenizer(
            question,
            context,
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

        start_token_idx = 0
        end_token_idx = 0

        try:
            # Since the data is exploded, there is only one span per row.
            char_spans = ast.literal_eval(location_str)
            if char_spans:
                span_str = char_spans[0]
                start_char, end_char = map(int, span_str.split())

                # Finds the start and end of the context within the tokenized sequence.
                context_token_start = 0
                while sequence_ids[context_token_start] != 1:
                    context_token_start += 1
                
                context_token_end = len(input_ids) - 1
                while sequence_ids[context_token_end] != 1:
                    context_token_end -= 1

                # Finds the token that contains the start of the answer.
                found_start = False
                for i in range(context_token_start, context_token_end + 1):
                    if offset_mapping[i][0] <= start_char and offset_mapping[i][1] >= start_char:
                        start_token_idx = i
                        found_start = True
                        break
                
                # Finds the token that contains the end of the answer.
                found_end = False
                for i in range(context_token_end, context_token_start - 1, -1):
                    if offset_mapping[i][0] <= end_char - 1 and offset_mapping[i][1] >= end_char - 1:
                        end_token_idx = i
                        found_end = True
                        break
                
                # If the span was not fully found (e.g., truncated), it is marked as unanswerable.
                if not (found_start and found_end and start_token_idx <= end_token_idx):
                    start_token_idx, end_token_idx = 0, 0
        except (ValueError, SyntaxError):
            # If the location is invalid or empty, it is an unanswerable question.
            start_token_idx, end_token_idx = 0, 0

        all_input_ids.append(torch.tensor(input_ids))
        all_attention_masks.append(torch.tensor(attention_mask))
        all_start_positions.append(torch.tensor(start_token_idx))
        all_end_positions.append(torch.tensor(end_token_idx))
        
        # Stores evaluation information.
        eval_info['offset_mapping'].append(offset_mapping)
        eval_info['sequence_ids'].append(sequence_ids)
        eval_info['contexts'].append(context)
        eval_info['answer_texts'].append(context[start_char:end_char] if 'start_char' in locals() and 'end_char' in locals() else "")

    dataset_items = [{
        'input_ids': all_input_ids[i],
        'attention_mask': all_attention_masks[i],
        'start_positions': all_start_positions[i],
        'end_positions': all_end_positions[i],
    } for i in range(len(all_input_ids))]

    return dataset_items, eval_info


class PatientNotesQADataset(Dataset):
    # Custom Dataset for patient notes question answering.
    def __init__(self, dataset_items):
        self.dataset_items = dataset_items

    def __getitem__(self, idx):
        return self.dataset_items[idx]

    def __len__(self):
        return len(self.dataset_items)

def create_model_and_tokenizer(model_name: str):
    # Creates an instance of the AutoModelForQuestionAnswering model and its corresponding tokenizer.
    print(f"Loading model and tokenizer for: {model_name}")
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def train_and_evaluate_epoch(model, train_dataloader, test_dataloader, optimizer, device, epoch_num, total_epochs):
    # Trains and evaluates the model for one epoch.
    model.train() # Sets model to training mode.
    total_train_loss = 0
    train_progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch_num+1}/{total_epochs} Training", leave=False)

    for batch in train_progress_bar:
        optimizer.zero_grad() # Clears previous gradients.
        # Moves all tensors in the batch to the device.
        batch = {k: v.to(device) for k, v in batch.items()}
        
        outputs = model(**batch)
        loss = outputs.loss
        total_train_loss += loss.item()
        loss.backward() # Backpropagates.
        optimizer.step() # Updates model parameters.
        train_progress_bar.set_postfix({'loss': loss.item()})

    avg_train_loss = total_train_loss / len(train_dataloader)
    print(f"Epoch {epoch_num+1} - Avg. Training Loss: {avg_train_loss:.4f}")

    model.eval() # Sets model to evaluation mode.
    total_test_loss = 0
    test_progress_bar = tqdm(test_dataloader, desc=f"Epoch {epoch_num+1}/{total_epochs} Evaluation", leave=False)
    with torch.no_grad(): # Disables gradient calculation for evaluation.
        for batch in test_progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_test_loss += loss.item()
            test_progress_bar.set_postfix({'test_loss': loss.item()})

    avg_test_loss = total_test_loss / len(test_dataloader)
    print(f"Epoch {epoch_num+1} - Avg. Test Loss: {avg_test_loss:.4f}")
    
    return avg_train_loss, avg_test_loss

def get_char_sets_from_location_str(location_str: str, pn_text_len: int) -> list[set[int]]:
    # Converts a location string into a list of character sets (spans).
    true_spans_char_sets = []
    try:
        char_spans_from_str = ast.literal_eval(location_str)
        for span_str in char_spans_from_str:
            parts = span_str.split()
            if len(parts) == 2:
                start_char, end_char = int(parts[0]), int(parts[1])
                start_char = max(0, start_char) # Ensures start is not negative.
                end_char = min(pn_text_len, end_char) # Ensures end is within text bounds.
                if start_char < end_char:
                    true_spans_char_sets.append(set(range(start_char, end_char))) # Creates a set of character indices.
    except (ValueError, SyntaxError):
        pass
    return true_spans_char_sets

def calculate_overall_example_jaccard(predicted_char_sets: list[set[int]], true_char_sets: list[set[int]]) -> float:
    # Calculates the Jaccard score for a single example based on predicted and true character sets.
    union_predicted_chars = set().union(*predicted_char_sets) if predicted_char_sets else set()
    union_true_chars = set().union(*true_char_sets) if true_char_sets else set()
    if not union_predicted_chars and not union_true_chars: return 1.0 # Returns perfect match if both are empty.
    intersection_of_unions = len(union_predicted_chars.intersection(union_true_chars))
    union_of_unions = len(union_predicted_chars.union(union_true_chars))
    return intersection_of_unions / union_of_unions if union_of_unions > 0 else 0.0 # Returns no overlap if one is empty.

def get_char_span_from_logits(start_logits, end_logits, offset_mapping, sequence_ids):
    # Converts token-level model outputs to a character-level span.
    # This avoids the faulty `context.find()` method.
    start_logits = start_logits.cpu().numpy()
    end_logits = end_logits.cpu().numpy()

    context_indices = [i for i, seq_id in enumerate(sequence_ids) if seq_id == 1]
    if not context_indices: return None, None

    start_scores = start_logits[context_indices]
    end_scores = end_logits[context_indices]
    
    start_candidates = np.argsort(start_scores)[::-1]
    end_candidates = np.argsort(end_scores)[::-1]

    for start_cand_idx in start_candidates[:20]:
        for end_cand_idx in end_candidates[:20]:
            start_tok_idx = context_indices[start_cand_idx]
            end_tok_idx = context_indices[end_cand_idx]

            if start_tok_idx <= end_tok_idx:
                start_char, _ = offset_mapping[start_tok_idx]
                _, end_char = offset_mapping[end_tok_idx]
                return start_char, end_char
                
    return None, None # No valid span found.

def calculate_jaccard_on_data(model, device, eval_info, tokenizer, max_len, description="Evaluating Jaccard"):
    # Calculates Jaccard score using pre-processed data for efficiency.
    model.eval()
    all_jaccards = []
    
    for i in tqdm(range(len(eval_info['contexts'])), desc=description):
        # Retrieves pre-processed information.
        row = eval_info['original_df'].iloc[i]
        context = eval_info['contexts'][i]
        offset_mapping = eval_info['offset_mapping'][i]
        sequence_ids = eval_info['sequence_ids'][i]
        
        question = str(row['feature_text'])
        true_location_str = str(row['location'])

        # Tokenizes again to create the tensor for the model input.
        inputs = tokenizer(
            question, context,
            max_length=max_len, padding="max_length", truncation="only_second", return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            start_logits = outputs.start_logits.squeeze(0)
            end_logits = outputs.end_logits.squeeze(0)

        pred_char_start, pred_char_end = get_char_span_from_logits(
            start_logits, end_logits, offset_mapping, sequence_ids
        )
        
        predicted_char_sets = []
        if pred_char_start is not None and pred_char_end is not None:
             predicted_char_sets.append(set(range(pred_char_start, pred_char_end)))

        true_char_sets = get_char_sets_from_location_str(true_location_str, len(context))
        jaccard_score = calculate_overall_example_jaccard(predicted_char_sets, true_char_sets)
        all_jaccards.append(jaccard_score)

    return np.mean(all_jaccards)


def save_performance_plots(history, model_friendly_name):
    # Saves plots of training and test loss/Jaccard score development.
    epochs_range = range(1, len(history['train_loss']) + 1)
    safe_filename = model_friendly_name.replace('/', '_')
    output_dir = 'plots'
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.plot(epochs_range, history['train_loss'], 'o-', label='Trainings-Loss', color='blue')
    plt.plot(epochs_range, history['test_loss'], 'o--', label='Test-Loss', color='cyan')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Loss Development: {model_friendly_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'{safe_filename}_loss_plot.png'))
    plt.close()
    print(f"\nLoss plot saved to '{os.path.join(output_dir, f'{safe_filename}_loss_plot.png')}'")

    plt.figure(figsize=(12, 6))
    plt.plot(epochs_range, history['train_jaccard'], 's-', label='Train-Jaccard Score', color='orange')
    plt.plot(epochs_range, history['test_jaccard'], 's--', label='Test-Jaccard Score', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Jaccard Score')
    plt.title(f'Jaccard Score Development: {model_friendly_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'{safe_filename}_jaccard_plot.png'))
    plt.close()
    print(f"Jaccard plot saved to '{os.path.join(output_dir, f'{safe_filename}_jaccard_plot.png')}'")


if __name__ == '__main__':
    RUN_IN_LOCAL_MODE = False
    DATA_PATH = "./" 
    MAX_LEN = 350
    EPOCHS = 5 

    MODELS_TO_TRAIN = [
        {
            'friendly_name': 'Bert-base',
            'model_path': 'bert-base-uncased',
            'learning_rate': 5e-5, 
            'batch_size': 8 
        },
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        features_df = pd.read_csv(os.path.join(DATA_PATH, "features.csv"))
        notes_df = pd.read_csv(os.path.join(DATA_PATH, "patient_notes.csv"))
        train_df_raw = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
    except FileNotFoundError as e:
        print(f"Error: Data not found. Ensure CSV files are in the directory '{DATA_PATH}'. Original error: {e}")
        exit()

    merged_df = train_df_raw.merge(features_df, on=['feature_num', 'case_num'], how='left') \
                            .merge(notes_df, on=['pn_num', 'case_num'], how='left')
    merged_df['location'] = merged_df['location'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    merged_df = merged_df.explode('location').reset_index(drop=True)
    merged_df['location'] = merged_df['location'].apply(lambda x: str([x]) if x else "[]")
    
    merged_df.dropna(subset=['pn_history', 'feature_text', 'location'], inplace=True)
    
    if RUN_IN_LOCAL_MODE:
        print("\n--- RUNNING IN LOCAL MODE ---")
        # Therefore, a smaller sample is used for the local run.
        df_to_use = merged_df.sample(n=200, random_state=42).copy()
    else:
        print("\n--- RUNNING IN FULL-DATA GPU MODE ---")
        df_to_use = merged_df.copy()

    print(f"Size of dataset being used: {len(df_to_use)} entries")
    
    for model_config in MODELS_TO_TRAIN:
        friendly_name = model_config['friendly_name']
        model_path = model_config['model_path']
        lr = model_config['learning_rate']
        batch_size = model_config['batch_size']

        print(f"\n\n{'='*20} TRAINING AND EVALUATING: {friendly_name.upper()} {'='*20}")
        print(f"Params: LR={lr}, Batch Size={batch_size}")

        train_data, test_data = train_test_split(df_to_use, test_size=0.2, random_state=42)
        print(f"Training data size: {len(train_data)}")
        print(f"Test data size: {len(test_data)}")

        current_model, current_tokenizer = create_model_and_tokenizer(model_path)
        current_model.to(device)

        # Pre-processes data once before the training loop for efficiency.
        print("Preprocessing training data...")
        train_items, train_eval_info = preprocess_data_for_qa(train_data, current_tokenizer, MAX_LEN)
        train_dataset = PatientNotesQADataset(train_items)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        print("Preprocessing test data...")
        test_items, test_eval_info = preprocess_data_for_qa(test_data, current_tokenizer, MAX_LEN)
        test_dataset = PatientNotesQADataset(test_items)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

        optimizer = optim.AdamW(current_model.parameters(), lr=lr)
        training_history = {'train_loss': [], 'test_loss': [], 'train_jaccard': [], 'test_jaccard': []}

        for epoch in range(EPOCHS):
            avg_train_loss, avg_test_loss = train_and_evaluate_epoch(current_model, train_dataloader, test_dataloader, optimizer, device, epoch, EPOCHS)
            training_history['train_loss'].append(avg_train_loss)
            training_history['test_loss'].append(avg_test_loss)

            # The pre-processed eval_info is used for Jaccard calculation.
            jaccard_score_epoch_train = calculate_jaccard_on_data(
                current_model, device, train_eval_info, current_tokenizer, MAX_LEN,
                description=f"Epoch {epoch+1}/{EPOCHS} Train Jaccard"
            )
            print(f"Epoch {epoch+1} - Train Jaccard: {jaccard_score_epoch_train:.4f}")
            training_history['train_jaccard'].append(jaccard_score_epoch_train)

            jaccard_score_epoch_test = calculate_jaccard_on_data(
                current_model, device, test_eval_info, current_tokenizer, MAX_LEN,
                description=f"Epoch {epoch+1}/{EPOCHS} Test Jaccard"
            )
            print(f"Epoch {epoch+1} - Test Jaccard: {jaccard_score_epoch_test:.4f}")
            training_history['test_jaccard'].append(jaccard_score_epoch_test)
        
        save_performance_plots(training_history, friendly_name)

        # Final Summary
        print(f"\n--- Final results for {friendly_name} ---")
        print(f"Final Test Jaccard: {training_history['test_jaccard'][-1]:.4f}")
        print(f"Best Test Jaccard: {max(training_history['test_jaccard']):.4f}")
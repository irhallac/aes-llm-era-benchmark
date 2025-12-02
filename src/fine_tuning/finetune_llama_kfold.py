"""
==================================================
PROJECT: Fine-Tuning LLMs for Multi-Trait Automatic Essay Assessment
AUTHOR: Ibrahim Riza Hallac
DATE: July 13, 2025

DESCRIPTION:
    -Fine-tunes pre-trained large language models (LLMs) for multi-trait essay scoring on the Feedback Prize - English Language Learning dataset.
    -Evaluates performance with metrics such as MAE and Quadratic Weighted Kappa (QWK).

DONE:
    -saving best model (optional)
    -k-fold cross-validation
    -gradient clipping
    -wandb logging
    -enhanced logging for all metrics (overall/per-feature QWK/MAE, averages/stds)
    -minimal config summary YAML

TO-DO:
    -Add results dictionary/CSV for comprehensive experiment tracking
    -Test on small data (e.g., stratified_100.csv) with various configurations
    -Automate multiple experiments (base/instruct models, prompt yes/no, strategies)
==================================================
"""

import argparse
import yaml
import os
import random
import numpy as np
import torch
import wandb
import pandas as pd
from datetime import datetime
from torch.optim import AdamW
from torch.nn import L1Loss
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils as utils  # For gradient clipping
import sys

from tqdm import tqdm;

# Load configuration
parser = argparse.ArgumentParser(description="Fine-tune LLaMA for AES scoring.")
parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config")
args = parser.parse_args()
with open(args.config, 'r') as file:
    config = yaml.safe_load(file)

# Configuration parameters
gpu = config['gpu']
train_data_path = config['train_data_path']
#test_data_path = config['test_data_path']
model_path = config['model']['path']
epochs = config['epochs']
batch_size = config['batch_size']
max_len = config['max_len']
prompt_path = config['prompt_path']
training_strategy = config.get('training_strategy', 'last_layer_only')
learning_rate = float(config['learning_rate'][training_strategy])
patience = config.get('patience', 15)
wandb_on = config['wandb_on']
use_prompt = config['use_prompt']
checkpoint_dir = config['checkpoint_dir']  # './FINE_TUNE/checkpoints/'
save_best_model = config.get('save_best_model', True)

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Extract keywords
model_short = os.path.basename(model_path).replace('Meta-', '').replace('-Instruct', '-I').replace('-', '_')
prompt_desc = 'prompt_yes' if use_prompt else 'prompt_no'
strategy_desc = training_strategy
if strategy_desc == 'top_n_layers':
    strategy_desc += f'_{config.get("layers_to_unfreeze", 0)}'

# Create descriptive folder name
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
folder_name = f"{model_short}_{strategy_desc}_{prompt_desc}_{current_time}"
timestamped_checkpoint_dir = os.path.join(checkpoint_dir, folder_name)
os.makedirs(timestamped_checkpoint_dir, exist_ok=True)

# Redirect all stdout/stderr to a log file inside the checkpoint directory
log_file_path = os.path.join(timestamped_checkpoint_dir, f"run_log.txt")
class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, message):
        for s in self.streams:
            s.write(message)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

sys.stdout = Tee(sys.stdout, open(log_file_path, 'w'))
sys.stderr = sys.stdout


used_config = {
    'config_path': args.config,
    'gpu': gpu,
    'train_data_path': train_data_path,
    'model_path': model_path,
    'epochs': epochs,
    'batch_size': batch_size,
    'max_len': max_len,
    'use_prompt': use_prompt,
    'prompt_path': prompt_path if use_prompt else None,
    'training_strategy': training_strategy,
    'layers_to_unfreeze': config.get('layers_to_unfreeze', None),
    'learning_rate': learning_rate,
    'k_folds': config['k_folds'],
    'wandb_on': wandb_on,
    'save_best_model': save_best_model
}
with open(os.path.join(timestamped_checkpoint_dir, 'used_config.yaml'), 'w') as f:
    yaml.dump(used_config, f)

if wandb_on:
    wandb.init(project="aes_llm_benchmark_public", entity="aes_paper1", name=f"run_{current_time}")
    wandb.config.update(config)
    print("wandb logging is enabled")
else:
    wandb.init(mode="disabled")
    print("wandb logging is disabled")

# Set seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

criterion = L1Loss(reduction='none')
feature_names = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]
valid_grades = torch.tensor([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
_model_path, _tokenizer_path = None, None

def get_model_and_tokenizer_paths(model_path):
    global _model_path, _tokenizer_path
    if os.path.exists(os.path.join(model_path, 'config.json')):
        _model_path = model_path
        _tokenizer_path = model_path
    else:
        print(f"No config.json found in {model_path}. Attempting to load from 'model' and 'tokenizer' subdirectories.")
        _model_path = os.path.join(model_path, 'model')
        _tokenizer_path = os.path.join(model_path, 'tokenizer')
    return _model_path, _tokenizer_path

model_path, tokenizer_path = get_model_and_tokenizer_paths(model_path)

class CustomModel(torch.nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.base = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=6)
        self.base.config.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token else tokenizer.eos_token_id

    def forward(self, input_ids, attention_mask):
        logits = self.base(input_ids=input_ids, attention_mask=attention_mask).logits
        return 1 + 4 * torch.sigmoid(logits)

def round_to_valid_grades(predictions):
    if predictions.dim() == 1:
        return torch.tensor([float(valid_grades[torch.argmin(torch.abs(valid_grades - pred))]) for pred in predictions])
    elif predictions.dim() == 2:
        return torch.stack([torch.tensor([float(valid_grades[torch.argmin(torch.abs(valid_grades - pred))]) for pred in sample]) for sample in predictions])
    else:
        raise ValueError("Predictions tensor must be 1D or 2D.")

from sklearn.metrics import cohen_kappa_score
def quadratic_weighted_kappa(y_true, y_pred):
    grade_to_bin = {float(grade): idx for idx, grade in enumerate(valid_grades)}
    y_true_bins = [grade_to_bin[float(val)] for val in y_true]
    y_pred_bins = [grade_to_bin[float(val)] for val in y_pred]
    if len(set(y_true_bins)) == 1 or len(set(y_pred_bins)) == 1:
        return 0.0
    qwk = cohen_kappa_score(y_true_bins, y_pred_bins, weights="quadratic")
    return qwk
    
class EssayDataset(Dataset):
    def __init__(self, dataframe, tokenizer, prompt_path=prompt_path, max_len=max_len, use_prompt=config['use_prompt']):
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.use_prompt = use_prompt
        self.prompt_template = self.load_prompt_template(prompt_path) if use_prompt else None

    def load_prompt_template(self, prompt_path):
        with open(prompt_path, "r") as file:
            return file.read()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        text = self.dataframe.loc[index, 'full_text']
        labels = self.dataframe.loc[index, feature_names].values.astype(float)
        # Conditionally apply the prompt or use raw text
        input_text = self.prompt_template.format(text=text) if self.use_prompt else text
        inputs = self.tokenizer(input_text, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt")
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': torch.tensor(labels, dtype=torch.float)
        }

def evaluate_test_with_per_qwk(model, dataloader, device):
    model.eval()
    total_mae = torch.zeros(len(feature_names), device=device)
    total_samples = 0
    all_true = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = (
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["labels"].to(device),
            )
            predictions = model(input_ids, attention_mask)
            rounded_predictions = round_to_valid_grades(predictions.cpu()).to(device)
            all_true.append(labels)
            all_preds.append(rounded_predictions)
            batch_mae = torch.abs(rounded_predictions - labels).mean(dim=0)
            total_mae += batch_mae * labels.size(0)
            total_samples += labels.size(0)

    all_true = torch.cat(all_true, dim=0).cpu()
    all_preds = torch.cat(all_preds, dim=0).cpu()
    total_qwk = quadratic_weighted_kappa(all_true.view(-1).tolist(), all_preds.view(-1).tolist())

    qwk_per_feature = []
    for i in range(len(feature_names)):
        qwk_per_feature.append(quadratic_weighted_kappa(all_true[:, i].tolist(), all_preds[:, i].tolist()))

    avg_mae = total_mae / total_samples
    return avg_mae.tolist(), total_qwk, qwk_per_feature

def set_training_strategy(model, training_strategy):
    training_strategy = training_strategy.strip().lower()
    print(f"Debug: training_strategy = '{training_strategy}'")
    total_layers = model.base.config.num_hidden_layers
    for name, param in model.named_parameters():
        param.requires_grad = False
        if training_strategy == 'last_layer_only':
            if "classifier" in name or "score" in name:
                param.requires_grad = True
        elif training_strategy == 'top_n_layers':
            layers_to_unfreeze = config.get('layers_to_unfreeze')
            if layers_to_unfreeze is None:
                raise ValueError("Must specify 'layers_to_unfreeze' in config for 'top_n_layers'")
            if "classifier" in name or "score" in name:
                param.requires_grad = True
            elif "layers" in name:
                layer_num = int(name.split("layers.")[1].split(".")[0])
                param.requires_grad = layer_num >= (total_layers - layers_to_unfreeze)
        elif training_strategy == 'all_layers':
            param.requires_grad = True
        elif training_strategy == 'prompt_tuning':
            pass
    if training_strategy not in ['last_layer_only', 'top_n_layers', 'all_layers', 'prompt_tuning']:
        print(f"Warning: Unknown training_strategy '{training_strategy}', defaulting to 'all_layers'")
        for name, param in model.named_parameters():
            param.requires_grad = True
    # Compute trainable layers and parameters consistently
    trainable_params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    trainable_layers = [name for name, _ in trainable_params]
    trainable_param_count = sum(param.numel() for _, param in trainable_params)
    print(f"Trainable layers: {trainable_layers}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Trainable parameters: {trainable_param_count}")
    print("  ^  ^    " * 10)


def train_without_kfold():
    # ... (keep the original function as is)
    pass

def log_cv_results(fold_results, timestamped_checkpoint_dir, feature_names):
    # Save per-fold results to CSV
    pd.DataFrame(fold_results).to_csv(os.path.join(timestamped_checkpoint_dir, 'fold_results.csv'), index=False, float_format='%.3f')

    # Compute averages and stds, rounded to 3 decimals
    avg_test_qwk = round(np.mean([r['Test QWK'] for r in fold_results]), 3)
    std_test_qwk = round(np.std([r['Test QWK'] for r in fold_results]), 3)
    avg_test_mae = round(np.mean([r['Test MAE Overall'] for r in fold_results]), 3)
    std_test_mae = round(np.std([r['Test MAE Overall'] for r in fold_results]), 3)
    avg_best_epoch = round(np.mean([r['Best Epoch'] for r in fold_results]), 1)

    per_feature_avg_mae = {fn: round(np.mean([r['Test MAE per Feature'][fn] for r in fold_results]), 3) for fn in feature_names}
    per_feature_std_mae = {fn: round(np.std([r['Test MAE per Feature'][fn] for r in fold_results]), 3) for fn in feature_names}
    per_feature_avg_qwk = {fn: round(np.mean([r['Test QWK per Feature'][fn] for r in fold_results]), 3) for fn in feature_names}
    per_feature_std_qwk = {fn: round(np.std([r['Test QWK per Feature'][fn] for r in fold_results]), 3) for fn in feature_names}

    # Write to txt (line format)
    txt_path = os.path.join(timestamped_checkpoint_dir, 'cv_results.txt')
    with open(txt_path, 'w') as f:
        f.write(f"Avg Best Epoch: {avg_best_epoch}\n")
        f.write(f"Avg Test QWK: {avg_test_qwk:.3f} ± {std_test_qwk:.3f}\n")
        f.write(f"Avg Test MAE: {avg_test_mae:.3f} ± {std_test_mae:.3f}\n")
        for fn in feature_names:
            f.write(f"{fn} Avg MAE: {per_feature_avg_mae[fn]:.3f} ± {per_feature_std_mae[fn]:.3f}\n")
            f.write(f"{fn} Avg QWK: {per_feature_avg_qwk[fn]:.3f} ± {per_feature_std_qwk[fn]:.3f}\n")

    # Write to Markdown table for readability
    md_path = os.path.join(timestamped_checkpoint_dir, 'cv_results.md')
    with open(md_path, 'w') as f:
        f.write("# CV Results\n\n")
        f.write(f"Avg Best Epoch: {avg_best_epoch}\n\n")
        f.write("| Metric | Overall | " + " | ".join(feature_names) + " |\n")
        f.write("|--------|---------|" + "--------|" * len(feature_names) + "\n")
        qwk_row = f"| Avg QWK | {avg_test_qwk:.3f} ± {std_test_qwk:.3f} | " + " | ".join(f"{per_feature_avg_qwk[fn]:.3f} ± {per_feature_std_qwk[fn]:.3f}" for fn in feature_names) + " |\n"
        mae_row = f"| Avg MAE | {avg_test_mae:.3f} ± {std_test_mae:.3f} | " + " | ".join(f"{per_feature_avg_mae[fn]:.3f} ± {per_feature_std_mae[fn]:.3f}" for fn in feature_names) + " |\n"
        f.write(qwk_row)
        f.write(mae_row)

    # Write to CSV for structured reporting (rows: metrics, columns: overall + features)
    data = {
        'Metric': ['Avg Best Epoch', 'Avg QWK', 'Std QWK', 'Avg MAE', 'Std MAE'],
        'Overall': [avg_best_epoch, avg_test_qwk, std_test_qwk, avg_test_mae, std_test_mae]
    }
    for fn in feature_names:
        data[fn] = [None, per_feature_avg_qwk[fn], per_feature_std_qwk[fn], per_feature_avg_mae[fn], per_feature_std_mae[fn]]
    pd.DataFrame(data).to_csv(os.path.join(timestamped_checkpoint_dir, 'cv_summary.csv'), index=False, float_format='%.3f')

def train_with_kfold():
    # Load full data for CV (assuming train_data_path is the full dataset; if not, concatenate with test_data_path)
    df = pd.read_csv(train_data_path)
    # If separate test, optionally: df = pd.concat([df, pd.read_csv(test_data_path)], ignore_index=True)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    kf = KFold(n_splits=config['k_folds'], shuffle=True, random_state=42)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        print(f"\nFold {fold+1}/{config['k_folds']}...")

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        train_loader = DataLoader(EssayDataset(train_df, tokenizer), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(EssayDataset(val_df, tokenizer), batch_size=batch_size)

        model = CustomModel(tokenizer).to(device)
        set_training_strategy(model, training_strategy)

        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.2*total_steps), num_training_steps=total_steps)

        best_qwk = -float('inf')
        best_epoch = 0
        best_val_mae = None
        best_val_qwk_per = None
        patience_counter = 0
        fold_dir = os.path.join(timestamped_checkpoint_dir, f'fold_{fold+1}')
        os.makedirs(fold_dir, exist_ok=True)
        best_model_path = os.path.join(fold_dir, 'best_model.pth')
        epoch_logs = []

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            batch_count = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            for batch in progress_bar:
                optimizer.zero_grad()
                preds = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
                loss_per_feature = criterion(preds, batch['labels'].to(device))
                loss = loss_per_feature.mean()
                loss.backward()
                utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                batch_count += 1
                # Optional: Update bar with current loss
                progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            progress_bar.close()        

            avg_train_loss = total_loss / batch_count

            val_mae, val_qwk, val_qwk_per = evaluate_test_with_per_qwk(model, val_loader, device)
            overall_val_mae = sum(val_mae) / len(val_mae)

            print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.3f}, Val MAE: {overall_val_mae:.3f}, Val QWK: {val_qwk:.3f}")

            epoch_logs.append({
                'Epoch': epoch + 1,
                'Train Loss': round(avg_train_loss, 3),
                'Val MAE': round(overall_val_mae, 3),
                'Val QWK': round(val_qwk, 3),
                'Val MAE per Feature': {fn: round(v, 3) for fn, v in zip(feature_names, val_mae)},
                'Val QWK per Feature': {fn: round(v, 3) for fn, v in zip(feature_names, val_qwk_per)}
            })

            if val_qwk > best_qwk:
                best_qwk = val_qwk
                best_val_mae = val_mae
                best_val_qwk_per = val_qwk_per
                best_epoch = epoch + 1
                patience_counter = 0
                if save_best_model:
                    torch.save(model.state_dict(), best_model_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        pd.DataFrame(epoch_logs).to_csv(os.path.join(fold_dir, 'epoch_logs.csv'), index=False)

        # Evaluate with best metrics
        if save_best_model:
            model.load_state_dict(torch.load(best_model_path))
            fold_mae, fold_qwk, fold_qwk_per = evaluate_test_with_per_qwk(model, val_loader, device)
        else:
            fold_mae = best_val_mae
            fold_qwk = best_qwk
            fold_qwk_per = best_val_qwk_per

        overall_fold_mae = sum(fold_mae) / len(fold_mae)

        fold_results.append({
            'Fold': fold + 1,
            'Best Epoch': best_epoch,
            'Epochs Run': epoch + 1,
            'Test QWK': round(fold_qwk, 3),
            'Test MAE Overall': round(overall_fold_mae, 3),
            'Test MAE per Feature': {fn: round(val, 3) for fn, val in zip(feature_names, fold_mae)},
            'Test QWK per Feature': {fn: round(val, 3) for fn, val in zip(feature_names, fold_qwk_per)}
        })

    # Log results using the function
    log_cv_results(fold_results, timestamped_checkpoint_dir, feature_names)

    print("K-Fold CV completed.")

if __name__ == "__main__":
    #train_without_kfold()
    train_with_kfold()
    print("Training completed and log saved.")

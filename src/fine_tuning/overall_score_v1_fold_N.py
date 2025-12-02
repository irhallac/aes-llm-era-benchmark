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

from tqdm import tqdm
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='Path to config file')
parser.add_argument('--target_fold', type=int, required=True, help='Fold number to run (1-based)')
args = parser.parse_args()

#c
# Example:
# python src/fine_tuning/overall_score_v1_fold_N.py --config configs/overall_debug.yaml --target_fold 1

# Extract "OE2" from "configs/config_OE2.yaml"
config_base = os.path.splitext(os.path.basename(args.config))[0]  # -> "config_OE2"
dataset_part = config_base.split("_", 1)[1] if "_" in config_base else config_base  # -> "OE2"

# Create dynamic folder name: "OE2_fold2"
dynamic_string = f"{dataset_part}_fold{args.target_fold}"
# Use it in your path

# Load the specified config file
with open(args.config, 'r') as file:
    config = yaml.safe_load(file)

# Configuration parameters
gpu = config['gpu']
train_data_path = config['train_data_path']
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
checkpoint_dir = config['checkpoint_dir']
save_best_model = config.get('save_best_model', True)

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("PyTorch sees", torch.cuda.device_count(), "GPU(s)")
print("CUDA available?", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
"""

model_short = os.path.basename(model_path).replace('Meta-', '').replace('-Instruct', '-I').replace('-', '_')
prompt_desc = 'prompt_yes' if use_prompt else 'prompt_no'
strategy_desc = training_strategy
if strategy_desc == 'top_n_layers':
    strategy_desc += f'_{config.get("layers_to_unfreeze", 0)}'
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
folder_name = f"{model_short}_{strategy_desc}_{prompt_desc}_{current_time}"
#timestamped_checkpoint_dir = os.path.join(checkpoint_dir, folder_name)
timestamped_checkpoint_dir = os.path.join(checkpoint_dir, dynamic_string, folder_name)
print(f"Creating directory: {timestamped_checkpoint_dir}")
os.makedirs(timestamped_checkpoint_dir, exist_ok=True)
print(f"Directory created: {timestamped_checkpoint_dir}")

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

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

criterion = L1Loss(reduction='none')

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
        self.base = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=1)
        self.base.config.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token else tokenizer.eos_token_id

    def forward(self, input_ids, attention_mask):
        logits = self.base(input_ids=input_ids, attention_mask=attention_mask).logits
        return 1 + 4 * torch.sigmoid(logits)  # Output: shape [batch, 1]

class EssayDataset(Dataset):
    def __init__(self, dataframe, tokenizer, prompt_path=prompt_path, max_len=max_len, use_prompt=config['use_prompt']):
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.use_prompt = use_prompt
        self.prompt_template = self.load_prompt_template(prompt_path) if use_prompt else None
        self.feature_names = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]

    def load_prompt_template(self, prompt_path):
        with open(prompt_path, "r") as file:
            return file.read()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        text = self.dataframe.loc[index, 'full_text']
        labels = self.dataframe.loc[index, self.feature_names].values.astype(float)
        labels = np.mean(labels)
        input_text = self.prompt_template.format(text=text) if self.use_prompt else text
        
        inputs = self.tokenizer(input_text, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt")
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': torch.tensor([labels], dtype=torch.float)
        }

from sklearn.metrics import cohen_kappa_score
def quadratic_weighted_kappa(y_true, y_pred):
    min_grade, max_grade = 1.0, 5.0
    n_grades = 9
    bins = np.linspace(min_grade, max_grade, n_grades)
    y_true_bins = np.digitize(y_true, bins) - 1
    y_pred_bins = np.digitize(y_pred, bins) - 1
    if len(set(y_true_bins)) == 1 or len(set(y_pred_bins)) == 1:
        return 0.0
    return cohen_kappa_score(y_true_bins, y_pred_bins, weights="quadratic")

def evaluate_overall(model, dataloader, device):
    model.eval()
    all_true = []
    all_preds = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device).squeeze(-1)
            preds = model(input_ids, attention_mask).squeeze(-1)
            all_true.append(labels.cpu())
            all_preds.append(preds.cpu())
    all_true = torch.cat(all_true, dim=0).numpy()
    all_preds = torch.cat(all_preds, dim=0).numpy()
    mae = np.mean(np.abs(all_preds - all_true))
    qwk = quadratic_weighted_kappa(all_true, all_preds)
    return mae, qwk

def log_cv_results(fold_results, timestamped_checkpoint_dir):
    pd.DataFrame(fold_results).to_csv(os.path.join(timestamped_checkpoint_dir, 'fold_results.csv'), index=False, float_format='%.3f')
    avg_qwk = round(np.mean([r['Test QWK'] for r in fold_results]), 3)
    std_qwk = round(np.std([r['Test QWK'] for r in fold_results]), 3)
    avg_mae = round(np.mean([r['Test MAE Overall'] for r in fold_results]), 3)
    std_mae = round(np.std([r['Test MAE Overall'] for r in fold_results]), 3)
    avg_best_epoch = round(np.mean([r['Best Epoch'] for r in fold_results]), 1)
    with open(os.path.join(timestamped_checkpoint_dir, 'cv_results.txt'), 'w') as f:
        f.write(f"Avg Best Epoch: {avg_best_epoch}\n")
        f.write(f"Avg Test QWK: {avg_qwk:.3f} ± {std_qwk:.3f}\n")
        f.write(f"Avg Test MAE: {avg_mae:.3f} ± {std_mae:.3f}\n")
    pd.DataFrame({
        'Metric': ['Avg Best Epoch', 'Avg QWK', 'Std QWK', 'Avg MAE', 'Std MAE'],
        'Overall': [avg_best_epoch, avg_qwk, std_qwk, avg_mae, std_mae]
    }).to_csv(os.path.join(timestamped_checkpoint_dir, 'cv_summary.csv'), index=False, float_format='%.3f')

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
    trainable_params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    trainable_layers = [name for name, _ in trainable_params]
    trainable_param_count = sum(param.numel() for _, param in trainable_params)
    print(f"Trainable layers: {trainable_layers}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Trainable parameters: {trainable_param_count}")
    print("  ^  ^    " * 10)


def train_with_kfold():
    target_fold = args.target_fold
    print(f"Running for Fold {target_fold} only.")

    df = pd.read_csv(train_data_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    kf = KFold(n_splits=config['k_folds'], shuffle=True, random_state=42)
    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        if fold + 1 != target_fold:  # Skip all folds except the target fold
            continue
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
        patience_counter = 0
        fold_dir = os.path.join(timestamped_checkpoint_dir, f'fold_{fold+1}')
        os.makedirs(fold_dir, exist_ok=True)
        best_model_path = os.path.join(fold_dir, 'best_model.pth')
        epoch_logs = []
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            batch_count = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False, disable=True)
            for batch in progress_bar:
                optimizer.zero_grad()
                preds = model(batch['input_ids'].to(device), batch['attention_mask'].to(device)).squeeze(-1)
                loss = criterion(preds, batch['labels'].to(device).squeeze(-1)).mean()
                loss.backward()
                utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                batch_count += 1
                progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            progress_bar.close()
            avg_train_loss = total_loss / batch_count
            val_mae, val_qwk = evaluate_overall(model, val_loader, device)
            print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.3f}, Val MAE: {val_mae:.3f}, Val QWK: {val_qwk:.3f}")
            epoch_logs.append({
                'Epoch': epoch + 1,
                'Train Loss': round(avg_train_loss, 3),
                'Val MAE': round(val_mae, 3),
                'Val QWK': round(val_qwk, 3)
            })
            if val_qwk > best_qwk:
                print(f"Epoch {epoch + 1}: Improved qwk {val_qwk}")
                best_qwk = val_qwk
                best_mae = val_mae
                best_epoch = epoch + 1
                patience_counter = 0
                if save_best_model:
                    torch.save(model.state_dict(), best_model_path)
            else:
                print(f"Epoch {epoch + 1}: No improvement, qwk {val_qwk}, patience {patience_counter + 1}/{patience}")
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
        pd.DataFrame(epoch_logs).to_csv(os.path.join(fold_dir, 'epoch_logs.csv'), index=False)
        if save_best_model:
            model.load_state_dict(torch.load(best_model_path))
            fold_mae, fold_qwk = evaluate_overall(model, val_loader, device)
        else:
            fold_mae = best_mae
            fold_qwk = best_qwk
        fold_results.append({
            'Fold': fold + 1,
            'Best Epoch': best_epoch,
            'Epochs Run': epoch + 1,
            'Test QWK': round(fold_qwk, 3),
            'Test MAE Overall': round(fold_mae, 3)
        })
    log_cv_results(fold_results, timestamped_checkpoint_dir)
    print("K-Fold CV completed.")

if __name__ == "__main__":
    import traceback
    try:
        train_with_kfold()
        print("Training completed and log saved.")
    except Exception as e:
        with open("error_log.txt", "a") as f:
            f.write(f"Error: {str(e)}\n")
            traceback.print_exc(file=f)
        raise

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, cohen_kappa_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
import torch
import torch.nn as nn
import torch.optim as optim
import time
import psutil
import os
from datetime import datetime
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# Simplified config (adapt paths as needed)
config = {
    'train_data_path': './data/train_set_kaggle.csv',
    'path_to_results': './ml_results/doc2vec',
    'k_folds': 2,
    'use_subset': True,
    'n_samples': 80
}

# Valid grades for rounding
valid_grades = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])

def round_to_valid_grades(predictions):
    """Rounds a 1D array of predictions to the nearest valid grade."""
    return np.array([valid_grades[np.argmin(np.abs(valid_grades - pred))] 
                    for pred in predictions])

def quadratic_weighted_kappa(y_true, y_pred):
    """
    Calculate QWK using scikit-learn's cohen_kappa_score.
    Assumes inputs are rounded to valid_grades.
    """
    grade_to_bin = {grade: idx for idx, grade in enumerate(valid_grades)}
    y_true_bins = [grade_to_bin[val] for val in y_true]
    y_pred_bins = [grade_to_bin[val] for val in y_pred]
    if len(set(y_true_bins)) == 1 or len(set(y_pred_bins)) == 1:
        return 0.0
    return cohen_kappa_score(y_true_bins, y_pred_bins, weights="quadratic")

start_time = time.time()

# Load Data
train_data = pd.read_csv(config['train_data_path'])
texts = train_data['full_text'].tolist()
labels_df = train_data[["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]].astype(float)

# Subset if enabled
if config['use_subset']:
    texts = texts[:config['n_samples']]
    labels_df = labels_df.iloc[:config['n_samples']]

print(f"Using {len(texts)} samples")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running ANN on {device}")

kf = KFold(n_splits=config['k_folds'], shuffle=True, random_state=42)

# ANN Model
class ANN(nn.Module):
    def __init__(self, input_dim):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Model dictionary
models = {
    "Linear Regression": LinearRegression(),
    "Support Vector Regression": SVR(),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
    "ANN": ANN
}

# Results storage
results = {name: {'MAE': [], 'QWK': [], 'Per Trait': {trait: {'MAE': [], 'QWK': []} for trait in labels_df.columns}} for name in models}
baseline_results = {'MAE': [], 'QWK': []}
best_fold, best_qwk, val_indices = {name: 0 for name in models}, {name: -1.0 for name in models}, None

for fold, (train_idx, val_idx) in enumerate(kf.split(texts), 1):
    # Prepare Doc2Vec model for this fold
    train_texts = [TaggedDocument(words=text.split(), tags=[i]) for i, text in enumerate(np.array(texts)[train_idx])]
    d2v_model = Doc2Vec(vector_size=100, window=5, min_count=2, workers=4, epochs=20)
    d2v_model.build_vocab(train_texts)
    d2v_model.train(train_texts, total_examples=d2v_model.corpus_count, epochs=d2v_model.epochs)

    # Generate embeddings for training and validation
    X_train = np.array([d2v_model.infer_vector(text.split()) for text in np.array(texts)[train_idx]])
    X_val = np.array([d2v_model.infer_vector(text.split()) for text in np.array(texts)[val_idx]])
    y_train_df = labels_df.iloc[train_idx]
    y_val_df = labels_df.iloc[val_idx]
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_train_tensor = torch.tensor(y_train_df.values, dtype=torch.float32, device=device)
    y_val_tensor = torch.tensor(y_val_df.values, dtype=torch.float32, device=device)

    print(f"\n🔄 Fold {fold}/{config['k_folds']}: Training size: {len(X_train)}, Validation size: {len(X_val)}")

    for model_name, model in models.items():
        trait_preds = []
        for trait_idx, trait in enumerate(labels_df.columns):
            if model_name == "ANN":
                ann_model = model(X_train.shape[1]).to(device)
                criterion = nn.L1Loss()
                optimizer = optim.Adam(ann_model.parameters(), lr=0.001)
                for epoch in range(20):
                    ann_model.train()
                    optimizer.zero_grad()
                    predictions = ann_model(X_train_tensor)
                    loss = criterion(predictions, y_train_tensor[:, trait_idx].unsqueeze(1))
                    loss.backward()
                    optimizer.step()
                ann_model.eval()
                with torch.no_grad():
                    raw_preds = ann_model(X_val_tensor).squeeze().cpu().numpy()
            else:
                model.fit(X_train, y_train_df[trait].values)
                raw_preds = model.predict(X_val)
            preds_rounded = round_to_valid_grades(raw_preds)
            truth_rounded = round_to_valid_grades(y_val_df[trait].values)
            mae_t = mean_absolute_error(truth_rounded, preds_rounded)
            qwk_t = quadratic_weighted_kappa(truth_rounded, preds_rounded)
            print(f"Fold {fold} - {model_name} - {trait}: MAE={mae_t:.3f}, QWK={qwk_t:.3f}")
            results[model_name]['Per Trait'][trait]['MAE'].append(mae_t)
            results[model_name]['Per Trait'][trait]['QWK'].append(qwk_t)
            trait_preds.append(preds_rounded)

        # Global metrics: flatten trait-major to sample-major
        preds_matrix = np.column_stack(trait_preds)
        all_preds = preds_matrix.flatten(order='C')
        truths_matrix = y_val_df.values
        all_truths = truths_matrix.flatten(order='C')
        global_mae = mean_absolute_error(all_truths, all_preds)
        global_qwk = quadratic_weighted_kappa(all_truths, all_preds)
        results[model_name]['MAE'].append(global_mae)
        results[model_name]['QWK'].append(global_qwk)
        print(f"[Fold {fold}] {model_name} Global MAE = {global_mae:.3f}, Global QWK = {global_qwk:.3f}")

        if global_qwk > best_qwk[model_name]:
            best_qwk[model_name], best_fold[model_name], val_indices = global_qwk, fold, val_idx

    # Baseline: predict mean of training samples' means
    sample_means = y_train_df.mean(axis=1).values
    baseline_preds = np.full_like(all_truths, sample_means.mean())
    baseline_preds_rounded = round_to_valid_grades(baseline_preds)
    baseline_results['MAE'].append(mean_absolute_error(all_truths, baseline_preds_rounded))
    baseline_results['QWK'].append(quadratic_weighted_kappa(all_truths, baseline_preds_rounded))
    print(f"Completed models and baseline for fold {fold}\n")

# Overall results
overall_data = [{'Model': name, 'Avg MAE': round(np.mean(results[name]['MAE']), 3), 
                 'MAE Std': round(np.std(results[name]['MAE']), 3), 
                 'Avg QWK': round(np.mean(results[name]['QWK']), 3), 
                 'QWK Std': round(np.std(results[name]['QWK']), 3)} for name in models]
overall_data.append({'Model': 'Baseline', 'Avg MAE': round(np.mean(baseline_results['MAE']), 3), 
                    'MAE Std': '-', 'Avg QWK': round(np.mean(baseline_results['QWK']), 3), 'QWK Std': '-'})
overall_df = pd.DataFrame(overall_data)

# Per-trait results
per_trait_data = []
for name in models:
    for trait in labels_df.columns:
        per_trait_data.append({
            'Model': name,
            'Trait': trait,
            'Avg MAE': round(np.mean(results[name]['Per Trait'][trait]['MAE']), 3),
            'Avg QWK': round(np.mean(results[name]['Per Trait'][trait]['QWK']), 3)
        })
per_trait_df = pd.DataFrame(per_trait_data)

# Reporting
print("\nOverall Results Table:")
print(overall_df)
print("\nPer-Trait Results Table:")
print(per_trait_df)
for name in models:
    print(f"\nOverall Average QWK ({name}): {overall_df[overall_df['Model'] == name]['Avg QWK'].iloc[0]:.3f}")
    for trait in labels_df.columns:
        print(f"{name} {trait} Avg MAE: {per_trait_df[(per_trait_df['Model'] == name) & (per_trait_df['Trait'] == trait)]['Avg MAE'].iloc[0]:.3f}, Avg QWK: {per_trait_df[(per_trait_df['Model'] == name) & (per_trait_df['Trait'] == trait)]['Avg QWK'].iloc[0]:.3f}")

# Save results to CSV with timestamp
os.makedirs(config['path_to_results'], exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
overall_df.to_csv(f"{config['path_to_results']}/overall_results_{timestamp}.csv", index=False, float_format='%.3f')
per_trait_df.to_csv(f"{config['path_to_results']}/per_trait_results_{timestamp}.csv", index=False, float_format='%.3f')

# Sample predictions from best fold
num_samples = 1
for name in models:
    print(f"\nSample Essay Predictions ({name}, fold {best_fold[name]}):")
    for idx, sample_idx in enumerate(val_indices[:num_samples]):
        snippet = texts[sample_idx][:200] + "..."
        print(f"\nEssay {idx+1} (Index {sample_idx}): {snippet}")
        print("True trait scores:")
        print(labels_df.iloc[sample_idx].to_dict())

# Resource Usage
end_time = time.time()
runtime = end_time - start_time
memory_mb = psutil.Process().memory_info().rss / (1024**2)
print(f"\nRuntime: {runtime:.2f}s, Memory: {memory_mb:.2f}MB")

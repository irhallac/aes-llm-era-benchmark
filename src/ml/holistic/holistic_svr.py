import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, cohen_kappa_score
from sklearn.svm import SVR
import time
import psutil  # For memory usage
import joblib
import os

# Simplified config (adapt paths as needed)
config = {
    'train_data_path': './data/train_set_kaggle.csv',
    'embeddings_path': './data/embeddings/bge_m3_embeddings.npz',
    'k_folds': 5,
    'use_subset': False,
    'n_samples': 100  # Configurable: Increase to 500 for more reliable metrics
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
    Assumes inputs are already rounded to valid_grades.
    """
    grade_to_bin = {grade: idx for idx, grade in enumerate(valid_grades)}
    y_true_bins = [grade_to_bin[val] for val in y_true]
    y_pred_bins = [grade_to_bin[val] for val in y_pred]
    if len(set(y_true_bins)) == 1 or len(set(y_pred_bins)) == 1:
        return 0.0
    return cohen_kappa_score(y_true_bins, y_pred_bins, weights="quadratic")

# Create models directory
os.makedirs('./models', exist_ok=True)

start_time = time.time()

# Load Data
train_data = pd.read_csv(config['train_data_path'])
texts     = train_data['full_text'].tolist()
labels_df = train_data[["cohesion", "syntax", "vocabulary", 
                        "phraseology", "grammar", "conventions"]].astype(float)

# Load embeddings
data       = np.load(config['embeddings_path'])
embeddings = data['embeddings']
assert len(embeddings) == len(train_data), "Embeddings and data length mismatch"

# Subset if enabled
if config['use_subset']:
    embeddings = embeddings[:config['n_samples']]
    texts      = texts[:config['n_samples']]
    labels_df  = labels_df.iloc[:config['n_samples']]

print(f"Using {len(embeddings)} samples")
print(f"Embedding size: {embeddings.shape[1]}")

kf = KFold(n_splits=config['k_folds'], shuffle=True, random_state=42)

# Results storage
results = {'Holistic-Trait MAE': [], 'Holistic-Trait QWK': []}
best_fold, best_qwk, val_indices = 0, -1.0, None

for fold, (train_idx, val_idx) in enumerate(kf.split(embeddings), 1):
    X_train, X_val = embeddings[train_idx], embeddings[val_idx]
    y_train_df     = labels_df.iloc[train_idx]
    y_val_df       = labels_df.iloc[val_idx]
    
    # Holistic-Trait evaluation (trained on mean of trait scores) with fresh SVR
    svr_holistic = SVR()
    svr_holistic.fit(X_train, y_train_df.mean(axis=1))
    predictions = svr_holistic.predict(X_val)
    predictions_rounded = round_to_valid_grades(predictions)
    y_val_rounded = round_to_valid_grades(y_val_df.mean(axis=1))

    holistic_mae = mean_absolute_error(y_val_rounded, predictions_rounded)
    holistic_qwk = quadratic_weighted_kappa(y_val_rounded, predictions_rounded)
    print(f"[Fold {fold}] Holistic-Trait MAE = {holistic_mae:.3f}, Holistic-Trait QWK = {holistic_qwk:.3f}")
    results['Holistic-Trait MAE'].append(holistic_mae)
    results['Holistic-Trait QWK'].append(holistic_qwk)

# Add Holistic-Trait results
holistic_df = pd.DataFrame({
    'Avg Holistic-Trait MAE': [np.mean(results['Holistic-Trait MAE'])],
    'Holistic-Trait MAE Std': [np.std(results['Holistic-Trait MAE'])],
    'Avg Holistic-Trait QWK': [np.mean(results['Holistic-Trait QWK'])],
    'Holistic-Trait QWK Std': [np.std(results['Holistic-Trait QWK'])]
})

print("\nHolistic-Trait Results Table:")
print(holistic_df)
print(f"Overall Average Holistic-Trait QWK: {holistic_df['Avg Holistic-Trait QWK'].iloc[0]:.4f}")

end_time = time.time()
runtime = end_time - start_time
memory_mb = psutil.Process().memory_info().rss / (1024**2)
print(f"\nRuntime: {runtime:.2f}s, Memory: {memory_mb:.2f}MB")

"""
(py3env_irh) [ec-ibrahimrh@c1-16 ml_src]$ python test_holistic.py 
Using 3911 samples
Embedding size: 2048
[Fold 1] Holistic-Trait MAE = 0.268, Holistic-Trait QWK = 0.708
[Fold 2] Holistic-Trait MAE = 0.266, Holistic-Trait QWK = 0.727
[Fold 3] Holistic-Trait MAE = 0.264, Holistic-Trait QWK = 0.707
[Fold 4] Holistic-Trait MAE = 0.283, Holistic-Trait QWK = 0.714
[Fold 5] Holistic-Trait MAE = 0.272, Holistic-Trait QWK = 0.714

Holistic-Trait Results Table:
   Avg Holistic-Trait MAE  Holistic-Trait MAE Std  Avg Holistic-Trait QWK  Holistic-Trait QWK Std
0                 0.27052                0.006642                0.714138                0.007182
Overall Average Holistic-Trait QWK: 0.7141

Runtime: 47.70s, Memory: 297.53MB
"""

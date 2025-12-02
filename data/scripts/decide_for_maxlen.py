import yaml
import pandas as pd
from transformers import AutoTokenizer
import matplotlib.pyplot as plt


def analyze_token_lengths(texts, tokenizer):
    """
    Analyze the distribution of token lengths for fine-tuning LLaMA or other transformer models.

    Parameters:
    - texts: List of strings, where each string is a document.
    - tokenizer: Tokenizer object to tokenize the texts.

    Outputs:
    - Histogram of token lengths.
    - Basic statistics (mean, max, percentiles).
    """
    print("Calculating token lengths...")
    # Calculate token lengths for each text
    token_lengths = [len(tokenizer(text, truncation=False)['input_ids']) for text in texts]

    # Basic statistics
    max_length = max(token_lengths)
    mean_length = sum(token_lengths) / len(token_lengths)
    percentiles = [50, 75, 90, 95, 99]
    percentile_values = {p: round(pd.Series(token_lengths).quantile(p / 100)) for p in percentiles}

    # Print statistics
    print(f"Maximum token length: {max_length}")
    print(f"Mean token length: {mean_length:.2f}")
    print("Percentile breakdown of token lengths:")
    for p, val in percentile_values.items():
        print(f"  {p}th percentile: {val}")

    # Plot histogram
    print("Plotting histogram...")
    plt.figure(figsize=(10, 6))
    plt.hist(token_lengths, bins=50, color='blue', alpha=0.7)
    plt.title("Token Length Distribution")
    plt.xlabel("Number of Tokens")
    plt.ylabel("Frequency")
    plt.axvline(mean_length, color='green', linestyle='dashed', linewidth=1, label="Mean")
    for p, val in percentile_values.items():
        plt.axvline(val, linestyle='dotted', label=f"{p}th percentile ({val})")
    plt.legend()
    plt.grid(True)
    plt.show()


# Load configuration
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Load training data
train_data_path = config['train_data_path']
model_path = config['model']['path']
tokenizer_path = f"{model_path}/tokenizer"

print("Loading training data...")
train_data = pd.read_csv(train_data_path)

# Extract the text column
text_column = "full_text"  # Change this if your text column has a different name
texts = train_data[text_column].tolist()

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# Analyze token lengths
print("Analyzing token lengths...")
analyze_token_lengths(texts, tokenizer)

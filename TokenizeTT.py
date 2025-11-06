from datasets import load_dataset
from transformers import GPT2TokenizerFast

# Load your dataset
data_files = {
    "train": "Dataset/train.json",
    "test": "Dataset/test.json"
}
dataset = load_dataset("json", data_files=data_files)

# Load the GPT2 tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("distilgpt2")

# GPT2 models don’t have a padding token by default — add one
tokenizer.pad_token = tokenizer.eos_token

# Define tokenization function
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=256  # You can increase if your responses are long
    )

# Apply tokenization to the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Save the tokenized datasets (optional but recommended)
tokenized_datasets["train"].save_to_disk("Dataset/tokenized_train")
tokenized_datasets["test"].save_to_disk("Dataset/tokenized_test")

print("✅ Tokenization complete!")
print(tokenized_datasets)

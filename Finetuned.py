from datasets import load_from_disk
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments
)
import torch
import os

# Load your tokenized datasets
train_dataset = load_from_disk("Dataset/tokenized_train")
test_dataset = load_from_disk("Dataset/tokenized_test")

print("✅ Tokenized datasets loaded successfully!")

# Load the base model and tokenizer
model_name = "distilgpt2"  # You can change to "gpt2" if needed
tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))

# Set up training configuration
training_args = TrainingArguments(
    output_dir="data/ModelOutput",
    logging_steps=100,
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=6,
    weight_decay=0.01,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    report_to="none",
)

# Add labels (for causal LM training, labels = input_ids)
train_dataset = train_dataset.map(lambda examples: {"labels": examples["input_ids"]})
test_dataset = test_dataset.map(lambda examples: {"labels": examples["input_ids"]})

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)
print("🚀 Starting training now...")

# Start fine-tuning
trainer.train()

# Save the fine-tuned model
os.makedirs("Finetuned_old/FineTuned_Model", exist_ok=True)
model.save_pretrained("Finetuned_old/FineTuned_Model")
tokenizer.save_pretrained("Finetuned_old/FineTuned_Model")

print("✅ Fine-tuning complete! Model saved at Finetuned_old/FineTuned_Model")

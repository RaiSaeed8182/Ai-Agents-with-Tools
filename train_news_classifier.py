"""
News Topic Classifier - BERT Fine-tuning
Fine-tunes BERT-base-uncased on AG News Dataset
"""

from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, f1_score
import torch
import numpy as np

# Load AG News Dataset
print("Loading AG News Dataset...")
dataset = load_dataset("ag_news")

# Labels: 1=World, 2=Sports, 3=Business, 4=Sci/Tech
label_names = ["World", "Sports", "Business", "Sci/Tech"]

# Load tokenizer and model
model_name = "bert-base-uncased"
print(f"Loading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=4
)

# Tokenize function
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

print("Tokenizing dataset...")
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]  # Remove original text, keep tokenized version
)

# Split dataset
train_dataset = tokenized_dataset["train"]
test_dataset = tokenized_dataset["test"]

# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Metrics function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {
        "accuracy": accuracy,
        "f1": f1
    }

# Training arguments
training_args = TrainingArguments(
    output_dir="./news_classifier_model",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train the model
print("Starting training...")
trainer.train()

# Evaluate
print("Evaluating on test set...")
results = trainer.evaluate()
print(f"Test Accuracy: {results['eval_accuracy']:.4f}")
print(f"Test F1-Score: {results['eval_f1']:.4f}")

# Save the model and tokenizer
print("Saving model...")
trainer.save_model("./news_classifier_model")
tokenizer.save_pretrained("./news_classifier_model")
print("Model saved successfully!")

print("\nTraining completed!")
print(f"Final Test Accuracy: {results['eval_accuracy']:.4f}")
print(f"Final Test F1-Score: {results['eval_f1']:.4f}")


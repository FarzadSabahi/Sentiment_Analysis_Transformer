import os
import torch
import random
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score, classification_report

# Set a random seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Load and preprocess your custom dataset (replace with your own dataset loading code)
# Example: 
# texts, labels = load_custom_dataset('your_dataset.csv')

# Split the dataset into train, validation, and test sets
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1
total_len = len(texts)
train_len = int(total_len * train_ratio)
val_len = int(total_len * val_ratio)
test_len = total_len - train_len - val_len

# Shuffle and split the data
indices = np.random.permutation(total_len)
train_indices = indices[:train_len]
val_indices = indices[train_len:train_len + val_len]
test_indices = indices[train_len + val_len:]

train_texts = [texts[i] for i in train_indices]
train_labels = [labels[i] for i in train_indices]

val_texts = [texts[i] for i in val_indices]
val_labels = [labels[i] for i in val_indices]

test_texts = [texts[i] for i in test_indices]
test_labels = [labels[i] for i in test_indices]

# Load a pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)

# Tokenize and encode the data
train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors='pt')
val_encodings = tokenizer(val_texts, truncation=True, padding=True, return_tensors='pt')
test_encodings = tokenizer(test_texts, truncation=True, padding=True, return_tensors='pt')

# Create PyTorch datasets
train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], torch.tensor(train_labels))
val_dataset = TensorDataset(val_encodings['input_ids'], val_encodings['attention_mask'], torch.tensor(val_labels))
test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], torch.tensor(test_labels))

# Create data loaders
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define training parameters
num_epochs = 5
learning_rate = 2e-5
warmup_steps = int(0.1 * num_epochs * len(train_loader))

# Create optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=learning_rate)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=len(train_loader) * num_epochs)

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
best_val_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
    
    # Calculate average training loss
    avg_train_loss = total_loss / len(train_loader)
    
    # Validation loop
    model.eval()
    val_predictions = []
    val_true_labels = []
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()
            logits = outputs.logits
            val_predictions.extend(logits.argmax(dim=1).tolist())
            val_true_labels.extend(labels.tolist())
    
    # Calculate average validation loss and accuracy
    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = accuracy_score(val_true_labels, val_predictions)
    
    print(f'Epoch {epoch + 1}/{num_epochs}:')
    print(f'Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

    # Save the model with the best validation loss
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_model.pth')

# Testing loop
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
test_predictions = []
test_true_labels = []
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        test_predictions.extend(logits.argmax(dim=1).tolist())
        test_true_labels.extend(labels.tolist())

# Calculate test accuracy and generate a classification report
test_accuracy = accuracy_score(test_true_labels, test_predictions)
classification_rep = classification_report(test_true_labels, test_predictions, target_names=class_labels)

print(f'Test Accuracy: {test_accuracy:.4f}')
print(f'Classification Report:\n{classification_rep}')

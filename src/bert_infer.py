import os
import argparse
import json
import datetime
import time
import torch
import pandas as pd
import numpy as np
from utils.helper import string_to_int
from transformers import BertModel, BertTokenizer, AdamW
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.metrics import roc_auc_score

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from config.config import Config

#####################################################################################################################
# Load configuration
config = Config(path="config/bert/")

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default=config.general.dataset)
parser.add_argument("--finetune_flag", default=config.general.finetune_flag, type=bool)

parser.add_argument("--model", default=config.eval_params.model)
parser.add_argument("--batch_size", default=config.eval_params.batch_size, type=int)

parser.add_argument("--seed", default=config.finetune.seed, type=int)
parser.add_argument("--numshot", default=config.finetune.numshot, type=int)
parser.add_argument("--epoch", default=config.finetune.epoch, type=int)
parser.add_argument("--lr", default=config.finetune.lr, type=float)
parser.add_argument("--patience", default=config.finetune.patience, type=int)

args = parser.parse_args()

# create res folder
path_save_model = f"eval_res/bert/{args.dataset}_numshot{args.numshot}/"
print()
print("Folder save: ", path_save_model)
if not os.path.exists(path_save_model):
    os.makedirs(path_save_model)
with open(path_save_model + 'commandline_args.txt', 'w') as f:
    json.dump(args.__dict__, f, indent=2)

#####################################################################################################################
class CommonLitDataset(Dataset):
        
    def get_tokens(self, texts):
        return self.tokenizer.batch_encode_plus(
            texts,
            truncation=True,
            max_length=256,
            padding='max_length',
        )
    
    def preprocess(self):
        self.df["final_text"] = self.df["instruction"].str.lower() + " [SEP] " + self.df["input"].str.lower()
        self.tokens = self.get_tokens(self.df["final_text"])
    
    def precompute(self, df):
        return get_tokens(df["text"])
    
    def __init__(self, df, tokenizer, mode="train"):
        self.mode = mode
        self.df = df
        self.tokenizer = tokenizer
        self.preprocess()

    def __len__(self):
        return len(self.df)

    def __getitem__(self ,idx):
        row = self.df.loc[idx]
        input_token = np.array(self.tokens["input_ids"][idx])
        mask = np.array(self.tokens["attention_mask"][idx])

        output = row["output"]
        target = np.array(output).astype(np.float32)
        return input_token, mask, target


class BERTModel(nn.Module):
    def __init__(self):
        super(BERTModel, self).__init__()
        self.bert = BertModel.from_pretrained(args.model, cache_dir="/hdd/")
        self.dropout = nn.Dropout(0.1)
        self.linear1 = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embedding = outputs.last_hidden_state
        x = torch.mean(embedding, dim=1)
        x = self.dropout(x)
        x = self.linear1(x)
        x = x.squeeze(1)
        x = torch.sigmoid(x)
        return x
#####################################################################################################################
#Load the tokenizer
tokenizer = BertTokenizer.from_pretrained(args.model, cache_dir="/hdd/")
BATCH_SIZE = args.batch_size

if args.finetune_flag:
    with open(f"serialized_data/{args.dataset}/non-private/serialized_train.json", 'r') as f:
        train_data = json.load(f)

    #Convert output string to integer
    train_data = string_to_int(train_data)
    traindf = pd.DataFrame.from_dict(train_data)

    shuffled = traindf.sample(frac=1, random_state=args.seed)
    train_size = args.numshot
    df_train = shuffled[:train_size].reset_index().drop(columns=["index"])
    train_set = CommonLitDataset(df_train, tokenizer)
    train_loader = DataLoader(train_set, batch_size = BATCH_SIZE, shuffle = True)

    df_valid = shuffled[train_size:].reset_index().drop(columns=["index"])
    valid_set = CommonLitDataset(df_valid, tokenizer)
    valid_loader = DataLoader(valid_set)

with open(f"serialized_data/{args.dataset}/non-private/serialized_test.json", 'r') as f:
    test_data = json.load(f)

test_data = string_to_int(test_data)
testdf = pd.DataFrame.from_dict(test_data).reset_index().drop(columns=["index"])
test_set = CommonLitDataset(testdf, tokenizer)
test_loader = DataLoader(test_set, batch_size = BATCH_SIZE)

#Training
model = BERTModel()
optimizer = AdamW(model.parameters(), lr=args.lr)
loss_fn = torch.nn.BCELoss()

start_time = time.time()

if args.finetune_flag:
    best_valid_loss = float('inf')  # Initialize best validation loss as infinity
    patience = args.patience  # Number of epochs to wait if validation loss doesn't decrease
    counter = 0  # Counter to keep track of epochs without improvement

    for epoch in range(args.epoch):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for batch_idx, (input_ids, attention_mask, labels) in enumerate(train_loader):
            optimizer.zero_grad()  # Zero the gradients

            # Forward pass
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)  # Calculate the loss

            # Backward pass and optimize
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            # Calculate running loss
            running_loss += loss.item()

            # Calculate accuracy
            predicted = (outputs > 0.5).float()  # Convert to binary predictions
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

            if batch_idx % 50 == 49:  # Print every 100 mini-batches
                print(f'Epoch {epoch + 1}, Batch {batch_idx + 1}, Loss: {running_loss / 100}')
                running_loss = 0.0

        # Calculate epoch-wise accuracy
        epoch_accuracy = correct_predictions / total_predictions
        print(f'Epoch {epoch + 1}, Accuracy: {epoch_accuracy}')
        
        # Validation phase
        model.eval()  # Set the model to evaluation mode
        valid_loss = 0.0
        
        with torch.no_grad():
            for val_batch_idx, (val_input_ids, val_attention_mask, val_labels) in enumerate(test_loader):
                # Forward pass
                val_outputs = model(val_input_ids, val_attention_mask)
                val_loss = loss_fn(val_outputs, val_labels)  # Calculate validation loss
                valid_loss += val_loss.item()

        # Average validation loss
        valid_loss /= len(valid_loader)
        print(f'Epoch {epoch + 1}, Validation Loss: {valid_loss}')

        # Early stopping
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            counter = 0  # Reset counter
            # Save the best model parameters
            best_model_params = model.state_dict()
        else:
            counter += 1
            if counter >= patience:
                print("Validation loss didn't decrease for {} epochs. Early stopping...".format(patience))
                break  # Exit training loop

    print('Finished Training')

    # Load the best model parameters
    model.load_state_dict(best_model_params)

model.eval()  # Set the model to evaluation mode
total_test_predictions = 0
correct_test_predictions = 0
test_predictions_proba = []
test_labels_list = []

with torch.no_grad():  # No need to calculate gradients during validation
    for test_batch_idx, (test_input_ids, test_attention_mask, test_labels) in enumerate(test_loader):
        # Forward pass
        test_outputs = model(test_input_ids, test_attention_mask)
        
        # Calculate validation accuracy
        test_predicted = (test_outputs > 0.5).float()  # Convert to binary predictions
        correct_test_predictions += (test_predicted == test_labels).sum().item()
        total_test_predictions += test_labels.size(0)
        
        # Calculate ROC AUC score
        test_predictions_proba.extend(test_outputs.cpu().numpy())  # Convert predictions to probabilities
        test_labels_list.extend(test_labels.cpu().numpy())

# Calculate validation accuracy
test_accuracy = correct_test_predictions / total_test_predictions
print(f'Test Accuracy: {test_accuracy}')

end_time = time.time()
infer_time = end_time - start_time

# Calculate ROC AUC score
roc_auc = roc_auc_score(test_labels_list, test_predictions_proba)
print(f'ROC AUC Score: {roc_auc}')

with open(path_save_model + "result.txt", "w") as f:
    f.write("Total time taken for inference: {:.2f} seconds".format(infer_time))
    f.write("\n")
    f.write("Test accuracy (%): " + str(100*test_accuracy))
    f.write("\n")
    f.write("Test ROC AUC score (%): " + str(100*roc_auc_score(test_labels_list, test_predictions_proba)))
import os
import argparse
import json
import datetime
import time
import random
import numpy as np
from retrying import retry
from utils.helper import convert_to_binary, calculate_classification_metrics
from mistralai import Mistral
from sklearn.metrics import accuracy_score, roc_auc_score

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from config.config import Config

#####################################################################################################################
# Load configuration
config = Config(path="config/mistral/")

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default=config.general.dataset)
parser.add_argument("--seed", default=config.general.seed, type=int)
parser.add_argument("--api_key", default=config.general.api_key)

parser.add_argument("--numshot", default=config.eval_params.numshot, type=int)
parser.add_argument("--model", default=config.eval_params.model)

args = parser.parse_args()

# Seed experiments
seed = args.seed
random.seed(seed)
np.random.seed(seed)

# create res folder
path_save_model = f"eval_res/mistral/{args.dataset}_numshot{args.numshot}/"
print()
print("Folder save: ", path_save_model)
if not os.path.exists(path_save_model):
    os.makedirs(path_save_model)
with open(path_save_model + 'commandline_args.txt', 'w') as f:
    json.dump(args.__dict__, f, indent=2)

#####################################################################################################################
def prompter(train_data, test_data, num_shots, seed=42):
    random.seed(seed)
    output = []
    sampled_idx = []
    
    selected_data = random.sample(train_data, num_shots)
    
    for entry in selected_data:
        sampled_idx.append(train_data.index(entry))
    
    for test_entry in test_data:
        prompt = test_entry["instruction"] + "\n\n###\nHere are some examples:\n\n"
        
        for example in selected_data:
            prompt += f"Input: {example['input']}\nOutput: {example['output']}\n\n"
        
        prompt += f"###\n\n<<<\nInput: {test_entry['input']}\n>>>"
        
        output.append({
            "input": prompt,
            "output": test_entry["output"]
        })
    
    return output, sampled_idx

client = Mistral(api_key=args.api_key)

@retry(wait_random_max=3000, stop_max_attempt_number = 10)
def mistral_infer(prompt):
    chat_response = client.chat.complete(
        model = args.model,
        messages = [
            {
                "role": "user",
                "content": prompt,
            },
        ]
    )
    return chat_response
#####################################################################################################################
#Load the dataset
with open(f"serialized_data/{args.dataset}/non-private/serialized_test.json", 'r') as f:
    test_data = json.load(f)
with open(f"serialized_data/{args.dataset}/non-private/serialized_train.json", 'r') as f:
    train_data = json.load(f)

print("Data loaded.")

pred = []
usages = []
total_prompt_tokens = 0
total_completion_tokens = 0
total_tokens = 0

#Start timer to track inference duration
start_time = time.time()

prompt_data, sampled_idx = prompter(train_data, test_data, args.numshot, seed=seed)

for i in range(len(prompt_data)):
    chat_response = mistral_infer(prompt_data[i]["input"])
    usages.append(chat_response.usage)
    total_prompt_tokens += chat_response.usage.prompt_tokens
    total_completion_tokens += chat_response.usage.completion_tokens
    total_tokens += chat_response.usage.total_tokens

    print(chat_response.choices[0].message.content)
    prompt_data[i]["model_output"] = chat_response.choices[0].message.content
    pred.append(convert_to_binary(chat_response.choices[0].message.content))

    with open(path_save_model + "output.json", 'w') as f:
        json.dump(prompt_data, f, indent=4)

    with open(path_save_model + "usage.txt", "a") as f:
        f.write(str(chat_response.usage))
        f.write("\n")

#End the timer
end_time = time.time()
infer_time = end_time - start_time

bin_source = np.load(f"serialized_data/{args.dataset}/non-private/binarized_X_test.npy")
np.save(path_save_model + "binarized_X_test.npy", bin_source)

np.save(path_save_model + "y_output.npy", pred)

with open(path_save_model + "result.txt", "w") as f:
    f.write("Total time taken for inference: {:.2f} seconds".format(infer_time))
    f.write("\n")
    f.write("Average time taken for an inference: {:.2f} seconds".format(infer_time/len(test_data)))
    f.write("\n")
    f.write("Total data: " + str(len(test_data)))
    f.write("\n")
    f.write("Total prompt tokens: " + str(total_prompt_tokens))
    f.write("\n")
    f.write("Total completion tokens: " + str(total_completion_tokens))
    f.write("\n")
    f.write("Total tokens: " + str(total_tokens))
    f.write("\n")
    f.write("Predicted 1s: " + str(sum(pred)))
    f.write("\n")
    f.write("Predicted 0s:" + str(sum(1 for num in pred if num == 0)))
    f.write("\n")

label = np.load(f"serialized_data/{args.dataset}/non-private/binarized_y_test.npy")
np.save(path_save_model + "binarized_y_test.npy", label)

tpr, tnr, fpr, fnr = calculate_classification_metrics(label, pred)

with open(path_save_model + 'result.txt', 'a') as f:
    f.write("Total actual 1s: " + str(sum(label)))
    f.write("\n")
    f.write("Total actual 0s:" + str(sum(1 for num in label if num == 0)))
    f.write("\n")
    f.write("Accuracy (%): " + str(100*accuracy_score(label, pred)))
    f.write("\n")
    f.write("ROC AUC score (%): " + str(100*roc_auc_score(label, pred)))
    f.write("\n")
    f.write("True Positive Rate (TPR): " + str(tpr))
    f.write("\n")
    f.write("True Negative Rate (TNR): " + str(tnr))
    f.write("\n")
    f.write("False Positive Rate (FPR): " + str(fpr))
    f.write("\n")
    f.write("False Negative Rate (FNR): " + str(fnr))
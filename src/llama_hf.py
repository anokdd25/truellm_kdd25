import os
import argparse
import json
import datetime
import time
import torch
import accelerate
import numpy as np
from utils.helper import convert_to_binary, prompter, extract_output, calculate_classification_metrics
from sklearn.metrics import accuracy_score, roc_auc_score
from transformers import AutoTokenizer, pipeline

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from config.config import Config

#####################################################################################################################
# Load configuration
config = Config(path="config/llama2/")

# Parse arguments
parser = argparse.ArgumentParser()

parser.add_argument("--dataset", default=config.general.dataset)

parser.add_argument("--model", default=config.model.model)
parser.add_argument("--device_map", default=config.model.device_map)

parser.add_argument("--top_k", default=config.infer.top_k, type=int)
parser.add_argument("--num_return_sequences", default=config.infer.num_return_sequences, type=int)
parser.add_argument("--max_length", default=config.infer.max_length, type=int)

args = parser.parse_args()

# create res folder
path_save_model = f"eval_res/llama2/{args.dataset}/"
print()
print("Folder save: ", path_save_model)
if not os.path.exists(path_save_model):
    os.makedirs(path_save_model)
with open(path_save_model + 'commandline_args.txt', 'w') as f:
    json.dump(args.__dict__, f, indent=2)

#Load inference data
with open(f"serialized_data/{args.dataset}/non-private/serialized_test.json", 'r') as f:
    eval_data = json.load(f)

#Load the pipeline
pipe = pipeline(
    task="text-generation",
    model=args.model,
    torch_dtype=torch.float16,
    device_map=args.device_map
)

tokenizer = AutoTokenizer.from_pretrained(args.model)

pred = []
start_time = time.time()

for d in eval_data:
    prompt = prompter(d)

    output = pipe(
        prompt,
        top_k=args.top_k,
        num_return_sequences=args.num_return_sequences,
        eos_token_id=tokenizer.eos_token_id,
        max_length=args.max_length
    )
    output = extract_output(output[0]['generated_text'])
    print(output)

    d['model_output'] = output
    pred.append(convert_to_binary(output))

end_time = time.time()
infer_time = end_time - start_time

#Save the inferences to a file
with open(path_save_model + "output.json", 'w') as f:
        json.dump(eval_data, f, indent=4)

bin_source = np.load(f"serialized_data/{args.dataset}/non-private/binarized_X_test.npy")
np.save(path_save_model + "binarized_X_test.npy", bin_source)

np.save(path_save_model + "y_output.npy", pred)

with open(path_save_model + "result.txt", "w") as f:
    f.write("Total time taken for inference: {:.2f} seconds".format(infer_time))
    f.write("\n")
    f.write("Average time taken for an inference: {:.2f} seconds".format(infer_time/len(eval_data)))
    f.write("\n")
    f.write("Total data: " + str(len(eval_data)))
    f.write("\n")
    f.write("Predicted 1s: " + str(sum(pred)))
    f.write("\n")
    f.write("Predicted 0s:" + str(sum(1 for num in pred if num == 0)))
    f.write("\n")

label = np.load(f"serialized_data/{args.dataset}/non-private/binarized_y_test.npy")

tpr, tnr, fpr, fnr = calculate_classification_metrics(label, pred)

np.save(path_save_model + "binarized_y_test.npy", label)

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

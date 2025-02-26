import os
import argparse
import json
import datetime
import time
import openai
import numpy as np
from openai import OpenAI
from retrying import retry
from utils.helper import calculate_classification_metrics
from sklearn.metrics import accuracy_score, roc_auc_score

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from config.config import Config

#####################################################################################################################
# Load configuration
config = Config(path="config/gpt/")

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default=config.general.dataset)
parser.add_argument("--api_key", default=config.general.api_key)

parser.add_argument("--model", default=config.eval_params.model)
parser.add_argument("--max_tokens", default=config.eval_params.max_tokens, type=int)
parser.add_argument("--temperature", default=config.eval_params.temperature, type=float)
parser.add_argument("--top_p", default=config.eval_params.top_p, type=float)

parser.add_argument("--numshot", default=config.finetune.numshot, type=int)
args = parser.parse_args()

# create res folder
path_save_model = f"eval_res/gpt3.5/{args.dataset}_numshot{args.numshot}/"
print()
print("Folder save: ", path_save_model)
if not os.path.exists(path_save_model):
    os.makedirs(path_save_model)
with open(path_save_model + 'commandline_args.txt', 'w') as f:
    json.dump(args.__dict__, f, indent=2)
    
#####################################################################################################################
client=OpenAI(api_key=args.api_key)    

@retry(wait_random_max=4000, stop_max_attempt_number = 5)
def infer(instruction, input):
    response = client.chat.completions.create(
        model = args.model,
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": input}
        ],
        max_tokens = args.max_tokens,
        temperature = args.temperature,
        top_p = args.top_p
    )
    return response

def convert_to_binary(string):
    if len(string) == 0:
        return 1
    # Check if the first character is '1' or '0'
    if string[0] == '1':
        return 1
    elif string[0] == '0':
        return 0
    
    # Check if '1' is present in the remaining part of the string
    if '1' in string:
        return 1
    else:
        return 0
#####################################################################################################################
#Load the test dataset
with open(f"serialized_data/{args.dataset}/non-private/serialized_test.json", 'r') as f:
    eval_data = json.load(f)

print("Evaluate data loaded.")

openai.api_key = args.api_key

pred = []
usages = []
total_prompt_tokens = 0
total_completion_tokens = 0
total_tokens = 0

#Start timer to track inference duration
start_time = time.time()

for d in eval_data:
    instruction = d['instruction']
    input = d['input']

    response = infer(instruction, input)

    usages.append(response.usage)
    total_prompt_tokens += response.usage.prompt_tokens
    total_completion_tokens += response.usage.completion_tokens
    total_tokens += response.usage.total_tokens

    print(response.choices[0].message.content)
    d['model_output'] = response.choices[0].message.content
    pred.append(convert_to_binary(d['model_output']))

    with open(path_save_model + "output.json", 'w') as f:
        json.dump(eval_data, f, indent=4)
    
    with open(path_save_model + "usage.txt", "a") as f:
        f.write(str(response.usage))
        f.write("\n")

    time.sleep(0.1)

#End the timer
end_time = time.time()
infer_time = end_time - start_time

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
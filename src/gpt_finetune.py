import argparse
import json
import random
from utils.helper import TrainingSetGenerator, class_balanced_serialized_sample
from openai import OpenAI

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

parser.add_argument("--seed", default=config.finetune.seed, type=int)
parser.add_argument("--numshot", default=config.finetune.numshot, type=int)
parser.add_argument("--model", default=config.finetune.model)

args = parser.parse_args()

seed = args.seed
random.seed(seed)

#Upload training dataset for finetuning to OpenAI client
#Load serialized data
with open(f"serialized_data/{args.dataset}/non-private/serialized_train.json", 'r') as f:
    data = json.load(f)

sampled_data = class_balanced_serialized_sample(data, args.numshot, seed=seed)
train_data = TrainingSetGenerator(sampled_data)

with open(f"finetune/gpt/{args.dataset}_numshot{args.numshot}.jsonl", "w") as f:
    for d in train_data:
        f.write(json.dumps(d) + '\n')

client = OpenAI(api_key=args.api_key)

openaifile = client.files.create(
    file=open(f"finetune/gpt/{args.dataset}_numshot{args.numshot}.jsonl", "rb"),
    purpose="fine-tune"
)

#####################################################################################################################
#Create finetuning job in OpenAI client
openaift = client.fine_tuning.jobs.create(
    training_file=openaifile.id,
    model=args.model
)

with open("finetune/gpt_ft.txt", "a") as f:
    f.write("\n")
    f.write(f"Dataset: {args.dataset}, Seed: {args.seed}, Train size: {args.train_size}, Train set path: {args.output_dir + args.filename}, Model: {args.model}")
    f.write("\n")
    f.write(str(openaift))
    f.write("\n")
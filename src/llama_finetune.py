import os
import json
import argparse
import random
import torch
from utils.helper import TrainingSetGenerator, class_balanced_serialized_sample
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer
from datasets import Dataset

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

parser.add_argument("--max_length", default=config.infer.max_length, type=int)

parser.add_argument("--seed", default=config.finetune.seed, type=int)
parser.add_argument("--numshot", default=config.finetune.numshot, type=int)
parser.add_argument("--per_device_train_batch_size", default=config.finetune.per_device_train_batch_size)
parser.add_argument("--gradient_accumulation_steps", default=config.finetune.gradient_accumulation_steps, type=int)
parser.add_argument("--learning_rate", default=config.finetune.learning_rate, type=float)
parser.add_argument("--logging_steps", default=config.finetune.logging_steps, type=int)
parser.add_argument("--num_train_epochs", default=config.finetune.num_train_epochs, type=int)
parser.add_argument("--max_steps", default=config.finetune.max_steps, type=int)
parser.add_argument("--save_steps", default=config.finetune.save_steps, type=int)
parser.add_argument("--save_total_limit", default=config.finetune.save_total_limit, type=int)
parser.add_argument("--gradient_checkpointing", default=config.finetune.gradient_checkpointing, type=bool)
parser.add_argument("--lora_r", default=config.finetune.lora_r, type=int)
parser.add_argument("--lora_alpha", default=config.finetune.lora_alpha, type=int)

args = parser.parse_args()

random.seed(args.seed)

output_dir = f"finetune/llama2/{args.dataset}_numshot{args.numshot}/"
print()
print("Folder save: ", output_dir)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
with open(output_dir + 'commandline_args.txt', 'w') as f:
    json.dump(args.__dict__, f, indent=2)


#Load the training data
with open(f"serialized_data/{args.dataset}/non-private/serialized_train.json", 'r') as f:
    data = json.load(f)

sampled_data = class_balanced_serialized_sample(data, args.numshot, seed=args.seed)
train_data = TrainingSetGenerator(sampled_data)
train_data = Dataset.from_list(train_data)

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=args.per_device_train_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    learning_rate=args.learning_rate,
    logging_steps=args.logging_steps,
    num_train_epochs=args.num_train_epochs,
    max_steps=args.max_steps,
    save_steps=args.save_steps,
    save_total_limit=args.save_total_limit,
    gradient_checkpointing=args.gradient_checkpointing,
)

peft_config = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    task_type="CAUSAL_LM",
)

#model = AutoModelForCausalLM.from_pretrained(args.model)
tokenizer = AutoTokenizer.from_pretrained(args.model)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype=torch.float16,
    #device_map=args.device_map
)
model.to(torch.device("cuda:2"))

trainer=SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    peft_config=peft_config,
    tokenizer=tokenizer
)
trainer.train()
trainer.save_model(output_dir)
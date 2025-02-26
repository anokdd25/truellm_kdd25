import argparse
import random
import os
import json
import joblib
import numpy as np
import pandas as pd
import time
import torch
from utils.helper import read_csv, DBEncoder, sample_generator, class_balanced_few_shot_sample
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

from tabpfn import TabPFNClassifier

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from config.config import Config

# Load configuration
config = Config(path="config/serialize/")

#####################################################################################################################
# Load configuration
config = Config(path="config/tabpfn/")

# Parse arguments
parser = argparse.ArgumentParser()

parser.add_argument("--dataset", default=config.general.dataset)
parser.add_argument("--seed", default=config.general.seed, type=int)
parser.add_argument("--device", default=config.general.device)
parser.add_argument("--private", default=config.general.private, type=bool)

parser.add_argument("--size", default=config.private.size, type=int)

parser.add_argument("--test_size", default=config.data.test_size, type=float)
parser.add_argument("--numshot", default=config.data.numshot, type=int)

args = parser.parse_args()

# Seed experiments
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# create res folder
if args.private:
    path_save_model = f"eval_res/tabpfn/{args.dataset}_{str(args.seed)}_numshot{str(args.numshot)}/private/"
else:
    path_save_model = f"eval_res/tabpfn/{args.dataset}_{str(args.seed)}_numshot{str(args.numshot)}/non-private/"
print()
print("Folder save: ", path_save_model)
if not os.path.exists(path_save_model):
    os.makedirs(path_save_model)
with open(path_save_model + 'commandline_args.txt', 'w') as f:
    json.dump(args.__dict__, f, indent=2)
print("Use Hardware : ", args.device)
print()

#####################################################################################################################
#Import data
X_df, y_df, f_df, label_pos = read_csv("dataset/" + args.dataset + "/" + args.dataset + ".csv",
                                       "dataset/" + args.dataset + "/" + args.dataset + ".info",
                                       shuffle=True)

db_enc = DBEncoder(f_df)
db_enc.fit(X_df, y_df)
X, y = db_enc.transform(X_df, y_df, normalized=True, keep_stat=True)
y = np.argmax(y, axis=1)

if args.private:
    print("Generating {} samples of {}".format(args.size, args.dataset))
    X_gen = sample_generator(X, db_enc, args.size, seed=seed)

    mean = np.array(db_enc.mean).reshape(1,-1) #Reshape the mean array to (1, cont_features)
    std = np.array(db_enc.std).reshape(1,-1) #Reshape the std array to (1, cont_features)
    reverted_continuous_values = X_gen[:, -db_enc.continuous_flen:] * std + mean #Revert the continuous values to pre-normalization
    cleaned_reverted_continuous_values = np.where(reverted_continuous_values < 0, 0, reverted_continuous_values) #Convert the negative values to 0
    cleaned_continuous_values = (cleaned_reverted_continuous_values - mean) / std #Normalize the cleaned continuous values

    X_gen[:, -db_enc.continuous_flen:] = cleaned_continuous_values #Replace the generated continuous values with the cleaned values


#####################################################################################################################
#Train TabPFN
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=seed)


if args.private:
    X_test_ori = X_test.copy()
    X_test = X_gen

    np.save(path_save_model + "binarized_X_original.npy", X_test_ori)
    np.save(path_save_model + "binarized_y_original.npy", y_test)

#Few shot sample the training data
X_shot, y_shot = class_balanced_few_shot_sample(X_train, y_train, args.numshot)
print(f"Sampled {sum(y_shot)} True and {len(y_shot) - sum(y_shot)} False data points.")

#####################################################################################################################
classifier = TabPFNClassifier(device=args.device)

start_time = time.time()
classifier.fit(X_shot, y_shot)
end_time = time.time()
train_time = end_time - start_time
print("Finished training: {} seconds".format(train_time))

start_time = time.time()
y_eval = classifier.predict(X_test)
end_time = time.time()
infer_time = (end_time - start_time)/len(y_eval)
with open(path_save_model + "performance.txt", "w") as f:
    f.write(f"Sampled {sum(y_shot)} True and {len(y_shot) - sum(y_shot)} False data points.")
    f.write("\n")
    f.write("Time taken for training: {} seconds".format(train_time))
    f.write("\n")
    f.write("Time taken for inference: {} seconds".format(infer_time))
    f.write("\n")
    f.write("Predicted 1s: " + str(sum(y_eval)))
    f.write("\n")
    f.write("Predicted 0s:" + str(sum(1 for num in y_eval if num == 0)))
    f.write("\n")    

np.save(path_save_model + "y_output.npy", y_eval)
np.save(path_save_model + "binarized_X.npy", X_test)

if not args.private:
    y_eval_proba = classifier.predict_proba(X_test)[:, 1]
    print("ROC AUC score (%): " + str(100*roc_auc_score(y_test, y_eval_proba)))

    with open(path_save_model + "performance.txt", "a") as f:
        f.write("Total actual 1s: " + str(sum(y_test)))
        f.write("\n")
        f.write("Total actual 0s:" + str(sum(1 for num in y_test if num == 0)))
        f.write("\n")
        f.write("Accuracy (%): " + str(100*accuracy_score(y_test, y_eval)))
        f.write("\n")
        f.write("ROC AUC score (%): " + str(100*roc_auc_score(y_test, y_eval_proba)))
        f.write("\n")
    
    np.save(path_save_model + "binarized_y.npy", y_test)
import argparse
import os
import json
import numpy as np
import pandas as pd
from utils.helper import read_csv, DBEncoder, sample_generator
from sklearn.model_selection import train_test_split

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from config.config import Config

# Load configuration
config = Config(path="config/serialize/")

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default=config.general.dataset)
parser.add_argument("--seed", default=config.general.seed, type=int)
parser.add_argument("--test_size", default=config.general.test_size, type=float)
parser.add_argument("--private", default=config.general.generate, type=bool)
parser.add_argument("--serialization_method", default=config.general.serialization_method)
parser.add_argument("--inflation", default=config.general.inflation, type=float)

parser.add_argument("--size", default=config.gen.size, type=int)
parser.add_argument("--clean", default=config.gen.clean, type=bool)

args = parser.parse_args()

# Seed experiments
seed = args.seed
np.random.seed(seed)

# create res folder
if args.private:
    path_save_model = f"serialized_data/{args.dataset}/private/"
else:
    path_save_model = f"serialized_data/{args.dataset}/non-private/"
print()
print("Folder save: ", path_save_model)
if not os.path.exists(path_save_model):
    os.makedirs(path_save_model)
with open(path_save_model + 'commandline_args.txt', 'w') as f:
    json.dump(args.__dict__, f, indent=2)
print()

#####################################################################################################################
#Load the data
print("Loading {} data".format(args.dataset))
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
    if args.clean:
        mean = np.array(db_enc.mean).reshape(1,-1) #Reshape the mean array to (1, cont_features)
        std = np.array(db_enc.std).reshape(1,-1) #Reshape the std array to (1, cont_features)
        reverted_continuous_values = X_gen[:, -db_enc.continuous_flen:] * std + mean #Revert the continuous values to pre-normalization
        cleaned_reverted_continuous_values = np.where(reverted_continuous_values < 0, 0, reverted_continuous_values) #Convert the negative values to 0
        cleaned_continuous_values = (cleaned_reverted_continuous_values - mean) / std #Normalize the cleaned continuous values

        X_gen[:, -db_enc.continuous_flen:] = cleaned_continuous_values
    X_gen_inv = db_enc.inverse_transform(X_gen)
    X_gen_df = pd.DataFrame(X_gen_inv)

#Data Preprocessing
with open("templates/preprocess.json", "r") as f:
    dict = json.load(f)

if args.dataset == 'adult':
    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation',
                    'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week',
                    'native_country']
    
    inflation = args.inflation

    def strip_string_columns(df):
        df[df.select_dtypes(['object']).columns] = df.select_dtypes(['object']).apply(lambda x: x.str.strip())

    if args.private:
        dataset = X_gen_df
    else:
        dataset = X_df

    dataset.columns = columns
    dataset = dataset.drop(columns=['fnlwgt', 'education_num'])
    #strip_string_columns(dataset)
    # Multiply all dollar columns with the inflation rate
    dataset[['capital_gain', 'capital_loss']] = (inflation * dataset[['capital_gain', 'capital_loss']]).astype(int)
    dataset[['age', 'hours_per_week']] = dataset[['age', 'hours_per_week']].astype(int)

    if args.private:
        dataset_train, dataset_test, X_train, X_test = train_test_split(dataset, X_gen, test_size=args.test_size, random_state=seed)
    
    else:
        dataset['label'] = y
        
        dataset_train, dataset_test, X_train, X_test, y_train, y_test = train_test_split(dataset, X, y, test_size=args.test_size, random_state=seed)

else:
    with open("dataset/" + args.dataset + "/columns.json", "r") as f:
        columns = json.load(f)

    if args.private:
        dataset = X_gen_df
    else:
        dataset = X_df

    #Convert int columns to int type
    for k in range(len(dataset.columns)):
        if f_df.shape[1] > 2 and f_df[2][k] == 'int':
            dataset[dataset.columns[k]] = dataset[dataset.columns[k]].apply(lambda x: str(int(x)) if pd.notna(x) else x)

    dataset.columns = columns

    dataset.replace(dict[args.dataset], inplace = True)

    if args.private:
        dataset_train, dataset_test, X_train, X_test= train_test_split(dataset, X_gen, test_size=args.test_size, random_state=seed)

    else:
        dataset["label"] = y
        dataset_train, dataset_test, X_train, X_test, y_train, y_test = train_test_split(dataset, X, y, test_size=args.test_size, random_state=seed)

#####################################################################################################################
def prompt_generator(generate, templates, template_num, dataset, dataset_name):
    data_dict = []

    data_template = templates[dataset_name]
    for t in data_template:
        if t["name"] == template_num:
            selected_template = t
    instruction = selected_template["instruction"]
    if "input" in selected_template:
        input = selected_template["input"]

    for _, row in dataset.iterrows():
        if "input" in selected_template:
            prompt = input.format(**row.to_dict()).replace('\n', '').replace('   ', '')
        else:
            prompt = ""
            for col_name, entry in row.items():
                if col_name != "label":
                    prompt += f"The {col_name} is {entry}. "

        if generate:
            data_dict.append({
                "instruction": instruction,
                "input": prompt
            })
        else:
            data_dict.append({
                "instruction": instruction,
                "input": prompt,
                "output": str(row['label'])
            })

    return data_dict

#####################################################################################################################
print("Serializing {} data with {} method".format(args.dataset, args.serialization_method))

with open("templates/prompt_template.json", 'r') as f:
    prompt_templates = json.load(f)

data_dict = prompt_generator(args.private, prompt_templates, args.serialization_method, dataset, args.dataset)
data_dict_train = prompt_generator(args.private, prompt_templates, args.serialization_method, dataset_train, args.dataset)
data_dict_test = prompt_generator(args.private, prompt_templates, args.serialization_method, dataset_test, args.dataset)

with open(path_save_model + "serialized_train.json", "w") as f:
    json.dump(data_dict_train, f, indent=4)

with open(path_save_model + "serialized_test.json", "w") as f:
    json.dump(data_dict_test, f, indent=4)

with open(path_save_model + "serialized_data.json", "w") as f:
    json.dump(data_dict, f, indent=4)

if args.private:
    np.save(path_save_model + "binarized_X.npy", X_gen)
    np.save(path_save_model + "binarized_X_train.npy", X_train)
    np.save(path_save_model + "binarized_X_test.npy", X_test)

else:
    np.save(path_save_model + "binarized_X.npy", X)
    np.save(path_save_model + "binarized_X_train.npy", X_train)
    np.save(path_save_model + "binarized_X_test.npy", X_test)

    np.save(path_save_model + "binarized_y.npy", y)
    np.save(path_save_model + "binarized_y_train.npy", y_train)
    np.save(path_save_model + "binarized_y_test.npy", y_test)

#dataset.to_csv(os.path.join(path_save_model, args.dataset + ".csv"), index=False)
#dataset_train.to_csv(os.path.join(path_save_model, args.dataset + "_train.csv"), index=False)
#dataset_test.to_csv(os.path.join(path_save_model, args.dataset + "_test.csv"), index=False)
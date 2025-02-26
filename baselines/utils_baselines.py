import ast
import warnings
warnings.filterwarnings('ignore')
import random
import os
import json
import numpy as np
import pandas as pd
import torch
import re

from src.utils.helper import read_info, read_csv, DBEncoder, class_balanced_few_shot_sample
from sklearn.model_selection import train_test_split
from aix360.algorithms.rbm import FeatureBinarizer, LogisticRuleRegression


# Load configuration

seed = 0


def set_seed(seed):
    # Seed experiments
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(dataset_name, dir, seed, private=False, num_shots=None, return_db_enc=False, data_root="./dataset"):

    X_df, y_df, f_df, label_pos = read_csv(os.path.join(data_root, dataset_name, f"{dataset_name}.csv"),
                                    os.path.join(data_root, dataset_name, f"{dataset_name}.info"),
                                    shuffle=True)
    db_enc = DBEncoder(f_df)
    db_enc.fit(X_df, y_df)

    if private:
        X_syn = np.load(dir + "/binarized_X.npy")
        y_syn = np.load(dir + "/y_output.npy")
        X_real = np.load(dir + "/binarized_X_original.npy")
        Y_real = np.load(dir + "/binarized_y_original.npy")
        X_train, X_test, Y_train, Y_test= train_test_split(X_syn, y_syn, test_size=0.2, random_state=seed)
        if return_db_enc:
            return X_train, X_real, X_test, Y_real, Y_train, Y_test, db_enc
        else:
            return X_train, X_real, X_test, Y_real, Y_train, Y_test
        
    else:
        if num_shots != None:
            X, y = db_enc.transform(X_df, y_df, normalized=True, keep_stat=True)
            y = np.argmax(y, axis=1)

            with open(os.path.join(data_root,dataset_name, f'{dataset_name}.columns'), 'w') as col_file:
                    for i, col in enumerate(db_enc.X_fname):
                        col_file.write(f'{i}\t{col}\n')
                    for j, col in enumerate(db_enc.y_fname):
                        col_file.write(f'{i+j+1}\t{col}\n')

            y_trainset = y
            X_train, X_test, Y_train, Y_test = train_test_split(X, y_trainset, test_size=0.2, stratify=y_trainset, random_state=seed)
            if num_shots != 'all':
                X_shot, Y_shot = class_balanced_few_shot_sample(X_train, Y_train, int(num_shots), seed)
            else: X_shot, Y_shot = X_train, Y_train
            if return_db_enc:
                return X_shot, X_test, None, Y_test, Y_shot, None, db_enc
            else:
                return X_shot, X_test, None, Y_test, Y_shot, None
        
        else:
            X = np.load(dir + "/binarized_X.npy")
            y = np.load(dir + "/binarized_y.npy")
            y_trainset = np.load(dir + "/y_output.npy")
            X_train, X_test, Y_train, Y_test, Y_trainset_train, Y_trainset_test = train_test_split(X, y, y_trainset, test_size=0.2, stratify=y_trainset, random_state=seed)

            if return_db_enc:
                return X_train, X_test, Y_train, Y_test, Y_trainset_train, Y_trainset_test, db_enc
            else:
                return X_train, X_test, Y_train, Y_test, Y_trainset_train, Y_trainset_test


def load_data_fb(dataset_name, dir, seed, private=False, num_shots=None, data_root='./dataset'):
    X_df, y_df, f_df, label_pos = read_csv(os.path.join(data_root, dataset_name, f"{dataset_name}.csv"),
                                                os.path.join(data_root, dataset_name, f"{dataset_name}.info"),
                                                shuffle=True)

    db_enc = DBEncoder(f_df)
    db_enc.fit(X_df, y_df)
    X, y = db_enc.transform(X_df, y_df, normalized=True, keep_stat=True)
    y = np.argmax(y, axis=1)

    fb = FeatureBinarizer(negations=True)
    # X = torch.cat((X_train, X_test), dim=0)
    TARGET_COLUMN = 'class_True'
    data_df = pd.DataFrame(X)
    data_df[TARGET_COLUMN] = y
    # data_df = data_df.drop(columns=[data_df.columns[-2]])
    fb.fit(data_df)

    if private:
        X_syn = np.load(dir + "/binarized_X.npy")
        Y_syn = np.load(dir + "/y_output.npy")
        X_real = np.load(dir + "/binarized_X_original.npy")
        Y_real = np.load(dir + "/binarized_y_original.npy")

        X_train, X_test, Y_train, Y_test = train_test_split(X_syn, Y_syn, test_size=0.2, random_state=seed)
        X_train_fb = fb.transform(pd.DataFrame(X_train))
        X_test_fb = fb.transform(pd.DataFrame(X_test))
        X_real_fb = fb.transform(pd.DataFrame(X_real))

        return X_train_fb, X_real_fb, X_test_fb, Y_real, Y_train, Y_test
    
    else:
        if num_shots != None:
            X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
            if num_shots != 'all':
                X_shot, Y_shot = class_balanced_few_shot_sample(X_train, Y_train, int(num_shots), seed)
            else: X_shot, Y_shot = X_train, Y_train

            X_shot_fb = fb.transform(pd.DataFrame(X_shot))
            X_test_fb = fb.transform(pd.DataFrame(X_test))

            return X_shot_fb, X_test_fb, None, Y_test, Y_shot, None

        else:
            X = np.load(dir + "/binarized_X.npy")
            y = np.load(dir + "/binarized_y.npy")
            y_trainset = np.load(dir + "/y_output.npy")
            X_train, X_test, Y_train, Y_test, Y_trainset_train, Y_trainset_test = train_test_split(X, y, y_trainset, test_size=0.2, stratify=y_trainset, random_state=seed)

            X_train_fb = fb.transform(pd.DataFrame(X_train))
            X_test_fb = fb.transform(pd.DataFrame(X_test))

            return X_train_fb, X_test_fb, Y_train, Y_test, Y_trainset_train, Y_trainset_test



def rules_aix360_to_human(model, dataset_name, save_path, seed, data_root="./dataset"):

    file_name = os.path.join(save_path, 'rules.txt')
    all_rules = ''
    index_to_colnames = colnames_index_dict(dataset_name, os.path.join(data_root, dataset_name))
    info = read_info(f"{data_root}/{dataset_name}/{dataset_name}.info")
    with open(f"{data_root}/{dataset_name}/columns.json", "r") as f:
        col = json.load(f)
    colnames_dict = {}
    for f in info[0][:-1]:
        colnames_dict[int(f[0])] = col[int(f[0])-1]

    _,_,_,_,_,_,db_enc = load_data(dataset_name, ".", seed, private=False, num_shots="all", return_db_enc=True)

        # Function to replace numbers with column names
    def replace_columns(rules, column_map):
        updated_rules = []
        for rule in rules:
            # Split the rule into parts
            parts = rule.split()
            # Replace the numbers with the corresponding column names
            updated_parts = [
                column_map.get(part, part) for part in parts
            ]
            # Join the parts back into a single string
            updated_rule = " ".join(updated_parts)
            updated_rules.append(updated_rule)

        return updated_rules

    def replace_int(rules, mean, std):
        updated_rules = []
        for rule in rules:
            parts = rule.split()

            for i in range(len(parts)):
                if parts[i] in list(mean.index):
                    parts[i+2] = "{:.2f}".format(float(parts[i+2]) * std[parts[i]] + mean[parts[i]])
            updated_rule = " ".join(parts)
            updated_rules.append(updated_rule)
        return updated_rules
    
    def replace_column_names(rules, mapping):
        # Define operators and the pattern to split clauses while retaining logical connectors
        operators = ["==", "!=", ">", "<", ">=", "<="]
        clause_pattern = r'(\sAND\s|\sOR\s)'
        
        updated_rules = []

        for rule in rules:
            # Split by logical connectors (AND, OR) to isolate each clause
            clauses = re.split(clause_pattern, rule)
            updated_clauses = []
            
            for clause in clauses:
                # Skip connectors (AND/OR)
                if clause.strip() in ["AND", "OR"]:
                    updated_clauses.append(clause)
                    continue
                
                # Split each clause by operators to identify left side and right side
                parts = re.split(r'(\s==\s|\s!=\s|\s>\s|\s<\s|\s>=\s|\s<=\s)', clause)
                if len(parts) < 3:
                    # Add clause as-is if it doesn't match the expected pattern
                    updated_clauses.append(clause)
                    continue

                # Extract left part (condition side), operator, and right part (value side)
                left_part, operator, right_part = parts[0].strip(), parts[1].strip(), parts[2].strip()
                
                # Check if left part contains an underscore or is a standalone integer
                if "_" in left_part:
                    prefix_str = left_part.split("_")[0]
                    if prefix_str.isdigit():
                        prefix = int(prefix_str)
                        if prefix in mapping:
                            suffix = left_part[left_part.index('_'):]  # Keep the suffix after "_"
                            left_part = mapping[prefix] + suffix
                elif left_part.isdigit():
                    prefix = int(left_part)
                    if prefix in mapping:
                        left_part = mapping[prefix]

                # Reassemble the updated clause
                updated_clause = f"{left_part} {operator} {right_part}"
                updated_clauses.append(updated_clause)
            
            # Join all updated clauses and connectors to form the final rule
            updated_rule = "".join(updated_clauses)
            updated_rules.append(updated_rule)
        
        return updated_rules

    if type(model) == LogisticRuleRegression:
        all_rules = model.explain()["rule"]
        coefficients = model.explain()["coefficient"]

        rules_with_colnames = replace_columns(all_rules, index_to_colnames)
        rules_with_original_continuous = replace_int(rules_with_colnames, db_enc.mean, db_enc.std)
        clean_rules = replace_column_names(rules_with_original_continuous, colnames_dict)
        # temp_filename = 'temp.csv'
        # all_rules.to_csv(temp_filename)
        # coefficients.to_csv('temp_coeff.csv')

        all_rules = ''
        for i, rule in enumerate(clean_rules):
            all_rules += str(coefficients[i]) + '\t' + rule + '\n'
    else:
        print("Not an explainer in : [LogisticRuleRegression]")
    # print(all_rules)
    with open(file_name, 'w') as file:
        file.write(str(all_rules))

    return all_rules



def colnames_index_dict(dataset_name, data_path='./dataset'):
    with open(os.path.join(data_path, dataset_name + ".columns"), 'r') as col_file:
        columns = col_file.readlines()
        columns = [[c.split('\t')[0], c.split('\t')[1]] for c in columns]

    index_to_colnames = {k: v.replace('\n', ' ') for k, v in columns}

    return index_to_colnames



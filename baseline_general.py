import warnings
from copy import copy
from baselines.utils_baselines import colnames_index_dict, load_data, load_data_fb, \
    rules_aix360_to_human, set_seed
import torch.nn as nn
import torch.optim as optim
warnings.filterwarnings('ignore')
import os
import numpy as np
import time
import torch
import sklearn
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn.metrics import accuracy_score, make_scorer, confusion_matrix, f1_score, mean_squared_error, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split, KFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from aix360.algorithms.rbm import FeatureBinarizer, LogisticRuleRegression, GLRM


SEEDS = [0, 1, 6, 7, 8]

def train_dnn_simple(model, X_train, Y_train, num_epochs=10):
    # Define the loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    batch_size = 256
    for epoch in range(num_epochs):
        # Shuffle the training data at the beginning of each epoch
        indices = torch.randperm(X_train.shape[0])
        X_train_shuffled = X_train[indices]
        y_train_shuffled = Y_train[indices]

        # Mini-batch training
        for i in range(0, X_train.shape[0], batch_size):
            batch_X = X_train_shuffled[i:i + batch_size]
            batch_y = y_train_shuffled[i:i + batch_size]

            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


def baseline_dnn(dir, verbose=False):
    trainaucs = []
    aucs = []
    accs = []
    tprs = []
    fprs = []
    f1s = []

    save_dir = os.path.join(dir, 'dnn')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for seed in SEEDS:
        set_seed(seed)

        save_dir_seed = os.path.join(save_dir, f'seed_{seed}')
        if not os.path.exists(save_dir_seed):
            os.mkdir(save_dir_seed)

        X = np.load(dir + "/binarized_X.npy")
        y = np.load(dir + "/binarized_y.npy")
        y_llm = np.load(dir + "/model_output.npy")
        X_train, _, Y_train, _ = train_test_split(X, y_llm, test_size=0.2, random_state=seed)
        _, X_test, _, Y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

        # Define the DNN model
        class DNN(nn.Module):
            def __init__(self, input_dim):
                super(DNN, self).__init__()
                self.fc1 = nn.Linear(input_dim, 16)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(16, 16)
                self.output = nn.Linear(16, 1)
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                x = self.fc1(x)
                x = self.relu(x)
                x = self.fc2(x)
                x = self.relu(x)
                x = self.output(x)
                x = self.sigmoid(x)
                return x

        # Create an instance of the DNN model
        model = DNN(X_train.shape[1])

        model = train_dnn_simple(model, X_train, Y_train, num_epochs=10)
        # Evaluation
        model.eval()
        with torch.no_grad():
            # Predict probabilities for the test data
            y_pred_proba = model(X_test).numpy()
            y_train_pred_proba = model(X_train).numpy()
        y_pred = (y_pred_proba >= 0.5).astype(int)
        # Calculate the AUC score
        auc = roc_auc_score(Y_test, y_pred_proba)
        trainauc = roc_auc_score(Y_train, y_train_pred_proba)
        # Calculate the confusion matrix
        conf_matrix = confusion_matrix(Y_test, y_pred)
        # Extract values from the confusion matrix
        true_negatives, false_positives, false_negatives, true_positives = conf_matrix.ravel()
        # Calculate TPR and FPR
        tpr = true_positives / (true_positives + false_negatives)
        fpr = false_positives / (false_positives + true_negatives)
        f1 = f1_score(Y_test, y_pred)
        tprs.append(tpr)
        fprs.append(fpr)
        f1s.append(f1)

        # Print the AUC score
        trainaucs.append(copy(trainauc))
        aucs.append(copy(auc))
        acc = accuracy_score(Y_test, y_pred)
        if verbose:
            print('dnn')
            print("Train AUC Score:", trainauc)
            print("AUC Score:", auc)
            print("Acc Score:", acc)
        accs.append(copy(acc))
        with open(os.path.join(save_dir_seed, 'metrics.txt'), 'w') as result_file:

            result_file.write(f'Accuracy: {acc}\n')
            result_file.write(f'Train AUC: {trainauc}\n')
            result_file.write(f'AUC: {auc}\n')
            result_file.write(f'FPR: {fpr}\n')
            result_file.write(f'TPR: {tpr}\n')
            result_file.write(f'F1-score: {f1}\n')

    return trainaucs, aucs, accs, tprs, fprs, f1s


def baseline_lr(dir, verbose=False):
    trainaucs = []
    aucs = []
    accs = []
    tprs = []
    fprs = []
    f1s = []
    save_dir = os.path.join(dir, 'lr')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for seed in SEEDS:
        set_seed(seed)

        save_dir_seed = os.path.join(save_dir, f'seed_{seed}')
        if not os.path.exists(save_dir_seed):
            os.mkdir(save_dir_seed)

        X = np.load(dir + "/binarized_X.npy")
        y = np.load(dir + "/binarized_y.npy")
        y_llm = np.load(dir + "/model_output.npy")
        X_train, X_llm_test, Y_train, Y_llm_test = train_test_split(X, y_llm, test_size=0.2, random_state=seed)
        _, X_test, _, Y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

        # Create a logistic regression model
        logreg_model = LogisticRegression()

        # Fit the model on the training data
        logreg_model.fit(X_train, Y_train)

        # Predict probabilities for the train data
        y_train_pred_proba = logreg_model.predict_proba(X_train)[:, 1]
        y_train_pred = np.argmax(logreg_model.predict_proba(X_train), axis=-1)

        # Predict probabilities for the test data
        y_pred_proba = logreg_model.predict_proba(X_test)[:, 1]
        y_pred = np.argmax(logreg_model.predict_proba(X_test), axis=-1)
        acc = accuracy_score(Y_test, y_pred)

        accs.append(copy(acc))
        # Calculate the AUC score
        trainauc = roc_auc_score(Y_train, y_train_pred_proba)
        auc = roc_auc_score(Y_test, y_pred_proba)
        # Calculate the confusion matrix
        conf_matrix = confusion_matrix(Y_test, y_pred)
        # Extract values from the confusion matrix
        true_negatives, false_positives, false_negatives, true_positives = conf_matrix.ravel()
        # Calculate TPR and FPR
        tpr = true_positives / (true_positives + false_negatives)
        fpr = false_positives / (false_positives + true_negatives)
        f1 = f1_score(Y_test, y_pred)
        trainaucs.append(copy(trainauc))
        aucs.append(copy(auc))
        tprs.append(tpr)
        fprs.append(fpr)
        f1s.append(f1)
        # Print the AUC score
        if verbose:
            print("\nlog ")
            print("AUC Score:", auc)
            print("Acc Score:", acc)
            print()

        with open(os.path.join(save_dir_seed, 'metrics.txt'), 'w') as result_file:

            result_file.write(f'Accuracy: {acc}\n')
            result_file.write(f'Train AUC: {trainauc}\n')
            result_file.write(f'AUC: {auc}\n')
            result_file.write(f'FPR: {fpr}\n')
            result_file.write(f'TPR: {tpr}\n')
            result_file.write(f'F1-score: {f1}\n')

    return trainaucs, aucs, accs, tprs, fprs, f1s


def baseline_decision_tree(dataset_name, dir, grid=True, verbose=False, private=False, num_shots=None):
    trainaucs = []
    aucs = []
    accs = []
    tprs = []
    fprs = []
    f1s = []
    rules_dic = []
    conds_dic = []
    save_dir = os.path.join(dir, 'decision_tree')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # index_to_colnames = colnames_index_dict(dataset, os.path.join('./dataset', dataset))

    for seed in SEEDS:
        set_seed(seed)

        save_dir_seed = os.path.join(save_dir, f'seed_{seed}')
        if not os.path.exists(save_dir_seed):
            os.mkdir(save_dir_seed)

        X_train, X_test, Y_train, Y_test, Y_trainset_train, Y_trainset_test = load_data(dataset_name, dir, seed, private,
                                                                                  num_shots)

        clf = DecisionTreeClassifier()

        if grid:
            # Define the parameter grid for the grid search
            param_grid = {
                "max_depth": np.random.randint(1, 21, 10),  # Integer between 1 and 20
                "min_samples_split": np.random.randint(2, 11, 10),  # Integer between 2 and 10
                "min_samples_leaf": np.random.randint(1, 11, 10),  # Integer between 1 and 10
                "max_features": ["sqrt", "log2", None] + list(np.random.uniform(0.1, 1.0, 5)),  # Feature selection
                "criterion": ["gini", "entropy"],  # Splitting criterion
            }
            # Perform grid search with cross-validation
            grid_search = RandomizedSearchCV(clf, param_grid, n_iter=30, cv=3)
            grid_search.fit(X_train, Y_trainset_train)

            # Retrieve the best model
            best_model = grid_search.best_estimator_
            # best_params = grid_search.best_params_
        else:
            clf.fit(X_train, Y_trainset_train)
            best_model = clf

        # Make prediction probabilities on train and test set
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        y_train_pred_proba = best_model.predict_proba(X_train)[:, 1]
        # Make predictions on train and test set
        y_pred = best_model.predict(X_test)
        y_train_pred = best_model.predict(X_train)
        
        if private:
            y_pred_proba_syn = best_model.predict_proba(Y_train)[:, 1]
            trainauc = roc_auc_score(Y_trainset_test, y_pred_proba_syn)

        # Calculate AUC (ROC)
        if num_shots != None:
            trainauc = np.nan
        elif not private:
            trainauc = roc_auc_score(Y_trainset_test, y_pred_proba)
        auc = roc_auc_score(Y_test, y_pred_proba)

        # Calculate accuracy
        accuracy = accuracy_score(Y_test, y_pred)

        # Calculate the confusion matrix
        conf_matrix = confusion_matrix(Y_test, y_pred)
        # Extract values from the confusion matrix
        true_negatives, false_positives, false_negatives, true_positives = conf_matrix.ravel()
        # Calculate TPR and FPR
        tpr = true_positives / (true_positives + false_negatives)
        fpr = false_positives / (false_positives + true_negatives)
        f1 = f1_score(Y_test, y_pred)
        tprs.append(tpr)
        fprs.append(fpr)
        f1s.append(f1)
        # Get the number of rules (number of nodes) in the best model
        num_conditions = best_model.tree_.node_count
        num_rules = sum(
            1 for i in range(num_conditions) if best_model.tree_.children_left[i] == best_model.tree_.children_right[i])
        conds_dic.append(num_conditions)
        trainaucs.append(copy(trainauc))
        aucs.append(copy(auc))
        accs.append(copy(accuracy))
        rules_dic.append(copy(num_rules))
        if verbose:
            # Print the results
            print("\nDecision tree")
            # print(best_params)
            print("Number of rules in the best model:", num_rules)
            print("AUC (ROC) score:", auc)
            print("Accuracy:", accuracy)
            print("Max depth of the tree: ", best_model.tree_.max_depth)

        text = sklearn.tree.export_text(best_model, max_depth=best_model.tree_.max_depth)
        lines = text.split('\n')

        if False:
            for i in range(len(lines)):
                l = lines[i]
                if 'feature' in l:
                    idx = l.index('_') + 1
                    feature_key = ''
                    char = l[idx]

                    while char != ' ':
                        feature_key += char
                        idx += 1
                        char = l[idx]
                    feature_key = str(int(feature_key) + 1)
                    lines[i] = l.replace(str(int(feature_key) - 1), index_to_colnames[feature_key]).replace('feature_',
                                                                                                            '') + '\n'

        with open(os.path.join(save_dir_seed, 'metrics.txt'), 'w') as result_file:

            result_file.write(f'Number of rules: {num_rules}\n')
            result_file.write(f'Number of conditions: {num_conditions}\n')
            result_file.write(f'Accuracy: {accuracy}\n')
            result_file.write(f'Train AUC: {trainauc}\n')
            result_file.write(f'AUC: {auc}\n')
            result_file.write(f'FPR: {fpr}\n')
            result_file.write(f'TPR: {tpr}\n')
            result_file.write(f'F1-score: {f1}\n')

        with open(os.path.join(save_dir_seed, 'rules.txt'), 'w') as result_file:

            result_file.writelines(lines)

    return trainaucs, aucs, accs, rules_dic, conds_dic, tprs, fprs, f1s


def baseline_rf(dataset_name, dir, grid=True, verbose=False, private=False, num_shots=None):
    trainaucs = []
    aucs = []
    accs = []
    tprs = []
    fprs = []
    f1s = []
    rules_dic = []
    conds_dic = []

    index_to_colnames = colnames_index_dict(dataset_name, os.path.join('./dataset', dataset_name))
    save_dir = os.path.join(dir, 'rf')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for seed in SEEDS:
        set_seed(seed)

        save_dir_seed = os.path.join(save_dir, f'seed_{seed}')
        save_dir_rules = os.path.join(save_dir_seed, 'rules')
        if not os.path.exists(save_dir_seed):
            os.mkdir(save_dir_seed)
        if not os.path.exists(save_dir_rules):
            os.mkdir(save_dir_rules)

        X_train, X_test, Y_train, Y_test, Y_trainset_train, Y_trainset_test = load_data(dataset_name, dir, seed, private,
                                                                                  num_shots)

        # Create a Random Forest classifier
        clf = RandomForestClassifier()

        if grid:
            # Define the parameter grid for the grid search
            param_grid = {
                "max_depth": [None, 2, 3, 4],
                "n_estimators": np.logspace(np.log10(10), np.log10(3000), num=10, dtype=int),
                "criterion": ["gini", "entropy"],
                "max_features": ["sqrt", "log2", None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                "min_samples_split": [2, 3],
                "min_samples_leaf": np.logspace(np.log10(1.5), np.log10(50.5), num=10, dtype=int),
            }

            # Perform grid search with cross-validation
            grid_search = RandomizedSearchCV(clf, param_grid, n_iter=30, cv=3)
            grid_search.fit(X_train, Y_trainset_train)

            # Retrieve the best model
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
        else:
            clf.fit(X_train, Y_trainset_train)
            best_model = clf
        # Make prediction probabilities on train and test set
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        y_train_pred_proba = best_model.predict_proba(X_train)[:, 1]
        # Make predictions on train and test set
        y_pred = best_model.predict(X_test)
        y_train_pred = best_model.predict(X_train)

        if private:
            y_pred_proba_syn = best_model.predict_proba(Y_train)[:, 1]
            trainauc = roc_auc_score(Y_trainset_test, y_pred_proba_syn)

        # Calculate AUC (ROC)
        if num_shots != None:
            trainauc = np.nan
        elif not private:
            trainauc = roc_auc_score(Y_trainset_test, y_pred_proba)
        auc = roc_auc_score(Y_test, y_pred_proba)
        # Calculate accuracy
        accuracy = accuracy_score(Y_test, y_pred)
        # Calculate the confusion matrix
        conf_matrix = confusion_matrix(Y_test, y_pred)
        # Extract values from the confusion matrix
        true_negatives, false_positives, false_negatives, true_positives = conf_matrix.ravel()
        # Calculate TPR and FPR
        tpr = true_positives / (true_positives + false_negatives)
        fpr = false_positives / (false_positives + true_negatives)
        f1 = f1_score(Y_test, y_pred)
        tprs.append(tpr)
        fprs.append(fpr)
        f1s.append(f1)
        # Get the number of trees in the best Random Forest model
        num_trees = best_model.n_estimators

        # Count the number of rules in each tree and calculate the total
        total_rules = 0
        total_conditions = 0
        for num_tree, tree in enumerate(best_model.estimators_):
            tree_structure = tree.tree_
            n_nodes = tree_structure.node_count
            n_leaves = sum(
                1 for i in range(n_nodes) if tree_structure.children_left[i] == tree_structure.children_right[i])
            total_conditions += n_nodes
            total_rules += n_leaves

            text = sklearn.tree.export_text(tree, max_depth=tree_structure.max_depth)
            # print(text)
            lines = text.split('\n')

            for i in range(len(lines)):
                l = lines[i]
                if 'feature' in l:
                    idx = l.index('_') + 1
                    feature_key = ''
                    char = l[idx]

                    while char != ' ':
                        feature_key += char
                        idx += 1
                        char = l[idx]
                    feature_key = str(int(feature_key) + 1)

                    lines[i] = l.replace(str(int(feature_key) - 1), index_to_colnames[feature_key]).replace('feature_',
                                                                                                            '') + '\n'

            with open(os.path.join(save_dir_rules, f'rules_tree_{num_tree}.txt'), 'w') as result_file:

                result_file.writelines(lines)

        rules_dic.append(copy(total_rules))
        conds_dic.append(total_conditions)
        trainaucs.append(copy(trainauc))
        aucs.append(copy(auc))
        accs.append(copy(accuracy))
        if verbose:
            # Print the results
            print('\nRandom Forest')
            # print(best_params)
            print("Number of trees in the best Random Forest model:", num_trees)
            # print("Total number of rules in the random forest:", total_rules)
            print("AUC (ROC) score:", auc)
            print("Accuracy:", accuracy)
            print("depth of each tree: ", [estimator.tree_.max_depth for estimator in best_model.estimators_])

        with open(os.path.join(save_dir_seed, 'metrics.txt'), 'w') as result_file:

            # result_file.write(f'Number of rules: {total_rules}\n')
            # result_file.write(f'Number of conditions: {total_conditions}\n')
            result_file.write(f'Accuracy: {accuracy}\n')
            result_file.write(f'Train AUC: {trainauc}\n')
            result_file.write(f'AUC: {auc}\n')
            result_file.write(f'FPR: {fpr}\n')
            result_file.write(f'TPR: {tpr}\n')
            result_file.write(f'F1-score: {f1}\n')

    return trainaucs, aucs, accs, rules_dic, conds_dic, tprs, fprs, f1s


def baseline_xgboost(dataset_name, dir, grid=True, verbose=False, private=False, num_shots=None):
    trainaucs = []
    aucs = []
    accs = []
    rules_dic = []
    conds_dic = []
    tprs = []
    fprs = []
    f1s = []

    index_to_colnames = colnames_index_dict(dataset_name, os.path.join('./dataset', dataset_name))
    save_dir = os.path.join(dir, 'xgboost')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for seed in SEEDS:
        set_seed(seed)

        save_dir_seed = os.path.join(save_dir, f'seed_{seed}')
        save_dir_rules = os.path.join(save_dir_seed, 'rules')
        if not os.path.exists(save_dir_seed):
            os.mkdir(save_dir_seed)
        if not os.path.exists(save_dir_rules):
            os.mkdir(save_dir_rules)

        X_train, X_test, Y_train, Y_test, Y_trainset_train, Y_trainset_test = load_data(dataset_name, dir, seed, private,
                                                                                  num_shots)

        # Create an XGBoost classifier
        xgb_model = xgb.XGBClassifier(n_estimators=15,
                                      max_depth=3,
                                      learning_rate=0.2,
                                      tree_method='gpu_hist',  # GPU-accelerated tree method
                                      predictor='gpu_predictor')  # Use GPU for predictions

        if grid:
            # Define the parameter grid for grid search
            param_grid = {
                "max_depth": np.random.randint(1, 12, 10),
                "n_estimators": np.random.randint(100, 6001, 10),
                "gamma": np.logspace(-8, 7, 10),
                "lambda": np.logspace(0, np.log10(4), 10),
                "alpha": np.logspace(-8, 2, 10),
            }

            # Perform grid search
            grid_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_grid, n_iter=30, scoring='roc_auc', cv=3)
            grid_search.fit(X_train, Y_trainset_train)

            # Get the best model and its parameters
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
        else:
            # xgb_model.fit(X_train, Y_train)
            best_model = xgb_model
            start_time = time.time()
            # Train the best model on the entire dataset
            best_model.fit(X_train, Y_trainset_train)
            end_time = time.time()
            train_time = end_time - start_time
            print(f"Training time: {train_time} seconds \n")

        # Predict probabilities and calculate ROC AUC score
        y_train_pred_proba = best_model.predict_proba(X_train)[:, 1]
        start_time = time.time()
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        end_time = time.time()
        infer_time = end_time - start_time
        print(f"Infer time: {infer_time} seconds \n")
        Y_pred = best_model.predict(X_test)

        if private:
            y_pred_proba_syn = best_model.predict_proba(Y_train)[:, 1]
            trainauc = roc_auc_score(Y_trainset_test, y_pred_proba_syn)

        if num_shots != None:
            trainauc = np.nan
        elif not private:
            trainauc = roc_auc_score(Y_trainset_test, y_pred_proba)
        roc_auc = roc_auc_score(Y_test, y_pred_proba)
        # Calculate the confusion matrix
        conf_matrix = confusion_matrix(Y_test, Y_pred)
        # Extract values from the confusion matrix
        true_negatives, false_positives, false_negatives, true_positives = conf_matrix.ravel()
        # Calculate TPR and FPR
        tpr = true_positives / (true_positives + false_negatives)
        fpr = false_positives / (false_positives + true_negatives)
        f1 = f1_score(Y_test, Y_pred)
        tprs.append(tpr)
        fprs.append(fpr)
        f1s.append(f1)

        # Get the number of trees in the best model
        if grid:
            num_trees = best_model.get_booster()
        else:
            num_trees = best_model.get_booster()
        # Get the text representation of the trees
        tree_text = best_model.get_booster().get_dump()

        # Count the number of rules in each tree and calculate the total
        total_rules = 0
        total_conds = 0
        for num_tree, tree in enumerate(tree_text):

            num_rules = tree.count('leaf')
            num_conditions = tree.count('[')
            total_conds += num_conditions
            total_rules += num_rules

            # print(text)
            lines = tree.split('\n')

            for i in range(len(lines)):
                l = lines[i]
                if '[f' in l:
                    idx = l.index('f') + 1
                    feature_key = ''
                    char = l[idx]

                    while char != '<' and char != '>':
                        feature_key += char
                        idx += 1
                        char = l[idx]

                    lines[i] = l.replace(str(int(feature_key) - 1), index_to_colnames[feature_key]).replace('f','') + '\n'

            with open(os.path.join(save_dir_rules, f'rules_tree_{num_tree}.txt'), 'w') as result_file:

                result_file.writelines(lines)

        accuracy = accuracy_score(Y_test, np.argmax(best_model.predict_proba(X_test), axis=-1))

        rules_dic.append(copy(total_rules))
        conds_dic.append(total_conds)
        trainaucs.append(copy(trainauc))
        aucs.append(copy(roc_auc))
        accs.append(copy(accuracy))
        if verbose:
            # Print the results
            print("\nXGboost")
            # print("Best Model Parameters:", best_params)
            print("ROC AUC Score:", roc_auc)
            print("Acc Score:", accuracy)
            print("Number of Trees:", num_trees)
            print("Total number of rules in the XGBoost model:", total_rules)

        with open(os.path.join(save_dir_seed, 'metrics.txt'), 'w') as result_file:

            result_file.write(f'Number of rules: {total_rules}\n')
            result_file.write(f'Number of conditions: {total_conds}\n')
            result_file.write(f'Accuracy: {accuracy}\n')
            result_file.write(f'Train AUC: {trainauc}\n') 
            result_file.write(f'AUC: {roc_auc}\n')
            result_file.write(f'FPR: {fpr}\n')
            result_file.write(f'TPR: {tpr}\n')
            result_file.write(f'F1-score: {f1}\n')

    return trainaucs, aucs, accs, rules_dic, conds_dic, tprs, fprs, f1s


def baseline_decision_models(dataset_name, dir, grid, models_to_train=('log', 'dt', 'rf', 'xgboost'), verbose=False,
                             private=False, num_shots=None):
    trainaucs, aucs, accs, rules_dic, conds_dic = {}, {}, {}, {}, {}
    tprs, fprs, f1s = {}, {}, {}
    if 'dnn' in models_to_train:
        print('Training DNN ...')
        trainaucs['dnn'], aucs['dnn'], accs['dnn'], tprs['dnn'], fprs['dnn'], f1s['dnn'] = baseline_dnn(dataset_name, verbose)
    if 'log' in models_to_train:
        print('Training logistic regression ...')
        trainaucs['log'], aucs['log'], accs['log'], tprs['log'], fprs['log'], f1s['log'] = baseline_lr(dataset_name, verbose)
    if 'dt' in models_to_train:
        print('Training Decision Tree ...')
        trainaucs['dt'], aucs['dt'], accs['dt'], rules_dic['dt'], conds_dic['dt'], tprs['dt'], fprs['dt'], f1s['dt'] = baseline_decision_tree(dataset_name, dir, grid, verbose, private, num_shots)
    if 'rf' in models_to_train:
        print('Training Random Forest ...')
        trainaucs['rf'], aucs['rf'], accs['rf'], rules_dic['rf'], conds_dic['rf'], tprs['rf'], fprs['rf'], f1s['rf'] = baseline_rf(dataset_name, dir, grid, verbose, private, num_shots)
    if 'xgboost' in models_to_train:
        print('Training XGBoost ...')
        trainaucs['xgboost'], aucs['xgboost'], accs['xgboost'], rules_dic['xgboost'], conds_dic['xgboost'], tprs['xgboost'], fprs['xgboost'], f1s['xgboost'] = baseline_xgboost(dataset_name, dir,
                                                                                                        grid, verbose, private, num_shots)

    return trainaucs, aucs, accs, rules_dic, conds_dic, tprs, fprs, f1s


def baseline_aix360(dataset_name, dir, models=('glrm',), verbose=False, explain=False, private=False, num_shots=None):
    trainaucs, aucs, accs, rules, conds = {k: [] for k in models}, {k: [] for k in models}, {k: [] for k in models}, {k: [] for k in models}, {k: [] for k
                                                                                                           in models}
    tprs, fprs, f1s = {k: [] for k in models}, {k: [] for k in models}, {k: [] for k in models}
    if 'glrm' in models:
        print('Training GLRM ...')
        trainaucs['glrm'], aucs['glrm'], accs['glrm'], rules['glrm'], conds['glrm'], tprs['glrm'], fprs['glrm'], f1s['glrm'] = baseline_glrm(dataset_name, dir, verbose, explain, private, num_shots)

    if verbose:
        for k in aucs.keys():
            print()
            print(k)

            print("Mean Train AUC: ", np.mean(trainaucs[k]))
            print("STD AUC: ", np.std(trainaucs[k]))

            print("Mean AUC: ", np.mean(aucs[k]))
            print("STD AUC: ", np.std(aucs[k]))

            print("Mean Acc: ", np.mean(accs[k]))
            print("Std Acc: ", np.std(accs[k]))

            print("Mean TPR: ", np.mean(tprs[k]))
            print("STD TPR: ", np.std(tprs[k]))

            print("Mean FPR: ", np.mean(fprs[k]))
            print("STD FPR: ", np.std(fprs[k]))

            print("Mean F1: ", np.mean(f1s[k]))
            print("STD F1: ", np.std(f1s[k]))

            print("conditions")
            print(np.mean(conds[k]))
            print(np.std(conds[k]))

            print("rules")
            print(np.mean(rules[k]))
            print(np.std(rules[k]))

    return trainaucs, aucs, accs, rules, conds, tprs, fprs, f1s


def baseline_glrm(dataset_name, dir, verbose=False, explain=False, private=False, num_shots=None):
    trainaucs = []
    aucs = []
    accs = []
    rules = []
    conds = []
    tprs = []
    fprs = []
    f1s = []

    save_dir = os.path.join(dir, 'glrm')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for seed in SEEDS:
        set_seed(seed)

        save_dir_seed = os.path.join(save_dir, f'seed_{seed}')
        if not os.path.exists(save_dir_seed):
            os.mkdir(save_dir_seed)

        X_train_fb, X_test_fb, Y_train, Y_test, Y_trainset_train, Y_trainset_test = load_data_fb(dataset_name, dir, seed,
                                                                                           private, num_shots)

        linear_model = LogisticRuleRegression(lambda0=0.01, lambda1=0.001)
        # print(X_train_fb.shape)X_test_fb
        start_time = time.time()
        linear_model.fit(X_train_fb, Y_trainset_train)
        end_time = time.time()
        train_time = end_time - start_time
        print(f"Training time: {train_time} seconds\n")

        Y_pred = linear_model.predict(X_test_fb)
        start_time = time.time()
        y_pred_proba = linear_model.predict_proba(X_test_fb)

        if private:
            y_pred_proba_syn = linear_model.predict_proba(Y_train)
            trainauc = roc_auc_score(Y_trainset_test, y_pred_proba_syn)

        end_time = time.time()
        infer_time = end_time - start_time
        print(f"Infer time: {infer_time} seconds\n")
        fpr, tpr, thresholds = metrics.roc_curve(Y_test, y_pred_proba)
        auc = metrics.auc(fpr, tpr)

        if num_shots != None:
            trainauc = np.nan
        elif not private:
            trainauc = roc_auc_score(Y_trainset_test, y_pred_proba)
        # Calculate the confusion matrix
        conf_matrix = confusion_matrix(Y_test, Y_pred)
        # Extract values from the confusion matrix
        true_negatives, false_positives, false_negatives, true_positives = conf_matrix.ravel()
        # Calculate TPR and FPR
        tpr = true_positives / (true_positives + false_negatives)
        fpr = false_positives / (false_positives + true_negatives)
        f1 = f1_score(Y_test, Y_pred)
        tprs.append(tpr)
        fprs.append(fpr)
        f1s.append(f1)
        if verbose:
            print("AUC", auc)
        if explain:
            rulestxt = rules_aix360_to_human(linear_model, dataset_name, save_dir_seed, seed)
        Rule = len(linear_model.explain()["rule"].to_numpy().tolist()) - 1
        condition = np.sum([x.count("AND") for x in linear_model.explain()["rule"].to_numpy().tolist()]) + Rule
        # print(Rule, condition)
        rules.append(Rule)
        conds.append(condition)
        trainaucs.append(trainauc)
        aucs.append(auc)
        accs.append(accuracy_score(Y_test, Y_pred))

        with open(os.path.join(save_dir_seed, 'metrics.txt'), 'w') as result_file:

            result_file.write(f'Number of rules: {Rule}\n')
            result_file.write(f'Number of conditions: {condition}\n')
            result_file.write(f'Accuracy: {accuracy_score(Y_test, Y_pred)}\n')
            result_file.write(f'Train AUC: {trainauc}\n')
            result_file.write(f'AUC: {auc}\n')
            result_file.write(f'FPR: {fpr}\n')
            result_file.write(f'TPR: {tpr}\n')
            result_file.write(f'F1-score: {f1}\n')

    return trainaucs, aucs, accs, rules, conds, tprs, fprs, f1s


def run_baselines(dataset_name, dir, models_to_train, grid, verbose=False, explain=True, private=False, num_shots=None):
    save_dir = os.path.join(dir)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    trainaucs, aucs, accs, rules_dic, conds_dic = {}, {}, {}, {}, {}
    tprs, fprs, f1s = {}, {}, {}
    trainaucs_dec, aucs_dec, accs_dec, rules_dec, conds_dec, tpr_dec, fpr_dec, f1_dec = baseline_decision_models(
        dataset_name, dir, grid, models_to_train['classic'],
        verbose, private, num_shots)
    trainaucs_aix, aucs_aix, accs_aix, rules_aix, conds_aix, tpr_aix, fpr_aix, f1_aix = baseline_aix360(dataset_name,
                                                                                                        dir,
                                                                                                        models_to_train[
                                                                                                            'aix'],
                                                                                                        verbose,
                                                                                                        explain,
                                                                                                        private,
                                                                                                        num_shots)

    for k in aucs_dec.keys():
        trainaucs[k] = trainaucs_dec[k]
        aucs[k] = aucs_dec[k]
        accs[k] = accs_dec[k]
        tprs[k] = tpr_dec[k]
        fprs[k] = fpr_dec[k]
        f1s[k] = f1_dec[k]
    for k in rules_dec.keys():
        rules_dic[k] = rules_dec[k]
        conds_dic[k] = conds_dec[k]

    for k in aucs_aix.keys():
        trainaucs[k] = trainaucs_aix[k]
        aucs[k] = aucs_aix[k]
        accs[k] = accs_aix[k]
        rules_dic[k] = rules_aix[k]
        conds_dic[k] = conds_aix[k]
        tprs[k] = tpr_aix[k]
        fprs[k] = fpr_aix[k]
        f1s[k] = f1_aix[k]

    for key, v in aucs.items():
        print('---' * 20)

        print(key)

        print("Mean Train AUC: ", np.mean(trainaucs[key]))
        print("STD Train AUC: ", np.std(trainaucs[key]))

        print("Mean AUC: ", np.mean(v))
        print("STD AUC: ", np.std(v))

        print("Mean Acc: ", np.mean(accs[key]))
        print("STD Acc: ", np.std(accs[key]))

        print("Mean TPR: ", np.mean(tprs[key]))
        print("STD TPR: ", np.std(tprs[key]))

        print("Mean FPR: ", np.mean(fprs[key]))
        print("STD FPR: ", np.std(fprs[key]))

        print("Mean F1: ", np.mean(f1s[key]))
        print("STD F1: ", np.std(f1s[key]))

        try:
            print("Mean rules: ", np.mean(rules_dic[key]))
            print("STD rules: ", np.std(rules_dic[key]))
        except:
            print("No rules for: ", key)

        try:
            print("Mean conditions: ", np.mean(conds_dic[key]))
            print("STD conditions: ", np.std(conds_dic[key]))
        except:
            print("No conditions for: ", key)

    with open(os.path.join(save_dir, f"baseline_results.txt"), 'w') as res_file:
        for key, v in aucs.items():
            res_file.write('\n' + '---' * 20 + '\n')

            res_file.write(key)
            res_file.write('\n\n')

            res_file.write(f"Mean Train AUC: {np.mean(trainaucs[key])}\n")
            res_file.write(f"STD Train AUC: {np.std(trainaucs[key])}\n")

            res_file.write(f"Mean AUC: {np.mean(v)}\n")
            res_file.write(f"STD AUC: {np.std(v)}\n")

            res_file.write(f"Mean Acc: {np.mean(accs[key])}\n")
            res_file.write(f"STD Acc: {np.std(accs[key])}\n")

            res_file.write(f"Mean TPR: {np.mean(tprs[key])}\n")
            res_file.write(f"STD TPR: {np.std(tprs[key])}\n")

            res_file.write(f"Mean FPR: {np.mean(fprs[key])}\n")
            res_file.write(f"STD FPR: {np.std(fprs[key])}\n")

            res_file.write(f"Mean F1: {np.mean(f1s[key])}\n")
            res_file.write(f"STD F1: {np.std(f1s[key])}\n")

            try:
                res_file.write(f"Mean rules: {np.mean(rules_dic[key])}\n")
                res_file.write(f"STD rules: {np.std(rules_dic[key])}\n")
            except:
                res_file.write(f"No rules for: {key}\n")

            try:
                res_file.write(f"Mean conditions: {np.mean(conds_dic[key])}\n")
                res_file.write(f"STD conditions: {np.std(conds_dic[key])}\n")
            except:
                res_file.write(f"No conditions for: {key}\n")

        res_file.write('\n\nTrain AUCS:' + str(trainaucs) + '\n')
        res_file.write('\n\nAUCS:' + str(aucs) + '\n')
        res_file.write('\n\nACC:' + str(accs) + '\n')
        res_file.write('\n\nTPR:' + str(tprs) + '\n')
        res_file.write('\n\nFPR:' + str(fprs) + '\n')
        res_file.write('\n\nF1:' + str(f1s) + '\n')
        res_file.write('\n\nRules:' + str(rules_dic) + '\n')
        res_file.write('\n\nConditions:' + str(conds_dic) + '\n')

    return aucs, accs, rules_dic, conds_dic, tprs, fprs, f1s


if __name__ == '__main__':

    grid = True
    verbose = False
    explain = True
    models_to_train = {'classic': ('dt', 'rf', 'xgboost'), 'aix': ('glrm')}

    datasets = ["heart"]  # ["adult", "bank", "blood", "calhousing", "creditg", "diabetes", "heart", "jungle"]
    seeds = [0]
    llm = 'tabpfn' # 'tabllm'
    llm_shots = [4] # [0, 4, 16, 64, 256]
    privacy = [True]
    num_shots = [None] # [None] or [4, 16, 64, 256, 'all']

    for dataset in datasets:
        for seed in seeds:
            for private in privacy:
                for llm_shot in llm_shots:
                    for num_shot in num_shots:

                        if private:
                            if llm == 'tabpfn':
                                datapath = f"eval_res/tabpfn/{dataset}_{seed}_numshot{llm_shot}/private/"
                            elif llm == 'tabllm':
                                datapath = f"eval_res/tabllm/t03b_{dataset}_numshot{llm_shot}_seed{seed}_ia3_pretrained100k/private/"
                        elif not private and num_shot == None:
                            if llm == 'tabpfn':
                                datapath = f"eval_res/tabpfn/{dataset}_{seed}_numshot{llm_shot}/non-private/"
                            elif llm == 'tabllm':
                                datapath = f"eval_res/tabllm/t03b_{dataset}_numshot{llm_shot}_seed{seed}_ia3_pretrained100k/non-private/"
                        elif not private and num_shot != None:
                            datapath = f"eval_res/baselines/{dataset}_numshot_{num_shot}/"
                        if not os.path.exists(datapath) and num_shot == None:
                            print("Skipping " + datapath)
                            continue
                        print("Running " + datapath)
                        try:
                            run_baselines(dataset, datapath, models_to_train, grid, verbose, explain, private, num_shot)
                        except ValueError as e:
                            if "at least 2 classes in the data" or "one class present in " in str(e):
                                print(f"Skipping due to single class error: {e}")
                            else:
                                raise
                        except IndexError as e:
                            # Handle index out of bounds error
                            if "index 1 is out of bounds for axis 1" in str(e):
                                print(f"Skipping due to index error caused by single class.")
                            else:
                                raise  # Re-raise for unexpected IndexError

    # run_baselines("adult", "eval_res/llm/gpt4_adult_numshot0/", models_to_train, grid, verbose, explain, private=False, num_shots=None)
import numpy as np
import pandas as pd
import random
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from scipy.stats import skewnorm


def read_info(info_path):
    with open(info_path) as f:
        f_list = []
        for line in f:
            tokens = line.strip().split()
            f_list.append(tokens)
        # print(f_list)
    return f_list[:-1], int(f_list[-1][-1])


def read_csv(data_path, info_path, shuffle=False):
    D = pd.read_csv(data_path, header=None)
    if shuffle:
        D = D.sample(frac=1, random_state=0).reset_index(drop=True)
    f_list, label_pos = read_info(info_path)
    f_df = pd.DataFrame(f_list)
    D.columns = f_df.iloc[:, 0]
    y_df = D.iloc[:, [label_pos]]
    X_df = D.drop(D.columns[label_pos], axis=1)
    f_df = f_df.drop(f_df.index[label_pos])
    return X_df, y_df, f_df, label_pos


class DBEncoder:
    """Encoder used for data discretization and binarization."""

    def __init__(self, f_df, discrete=False, y_one_hot=True, drop='first'):
        self.f_df = f_df
        self.discrete = discrete
        self.y_one_hot = y_one_hot
        self.label_enc = preprocessing.OneHotEncoder(categories='auto') if y_one_hot else preprocessing.LabelEncoder()
        self.feature_enc = preprocessing.OneHotEncoder(categories='auto', drop=drop)
        self.imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        self.X_fname = None
        self.y_fname = None
        self.discrete_flen = None
        self.continuous_flen = None
        self.mean = None
        self.std = None

    def split_data(self, X_df):
        discrete_data = X_df[self.f_df.loc[self.f_df[1] == 'discrete', 0]]
        continuous_data = X_df[self.f_df.loc[self.f_df[1] == 'continuous', 0]]
        if not continuous_data.empty:
            continuous_data = continuous_data.replace(to_replace=r'.*\?.*', value=np.nan, regex=True)
            continuous_data = continuous_data.astype(np.float64)
        return discrete_data, continuous_data

    def fit(self, X_df, y_df):
        X_df = X_df.reset_index(drop=True)
        y_df = y_df.reset_index(drop=True)
        discrete_data, continuous_data = self.split_data(X_df)
        self.label_enc.fit(y_df)
        self.y_fname = list(self.label_enc.get_feature_names_out(y_df.columns)) if self.y_one_hot else y_df.columns

        if not continuous_data.empty:
            # Use mean as missing value for continuous columns if do not discretize them.
            self.imp.fit(continuous_data.values)
        if not discrete_data.empty:
            # One-hot encoding
            self.feature_enc.fit(discrete_data)
            feature_names = discrete_data.columns
            self.X_fname = list(self.feature_enc.get_feature_names_out(feature_names))
            self.discrete_flen = len(self.X_fname)
            if not self.discrete:
                self.X_fname.extend(continuous_data.columns)
        else:
            self.X_fname = continuous_data.columns
            self.discrete_flen = 0
        self.continuous_flen = continuous_data.shape[1]

    def transform(self, X_df, y_df, normalized=False, keep_stat=False):
        X_df = X_df.reset_index(drop=True)
        y_df = y_df.reset_index(drop=True)
        discrete_data, continuous_data = self.split_data(X_df)
        # Encode string value to int index.
        y = self.label_enc.transform(y_df.values.reshape(-1, 1))
        if self.y_one_hot:
            y = y.toarray()

        if not continuous_data.empty:
            # Use mean as missing value for continuous columns if we do not discretize them.
            continuous_data = pd.DataFrame(self.imp.transform(continuous_data.values),
                                           columns=continuous_data.columns)
            if normalized:
                if keep_stat:
                    self.mean = continuous_data.mean()
                    self.std = continuous_data.std()
                continuous_data = (continuous_data - self.mean) / self.std
        if not discrete_data.empty:
            # One-hot encoding
            discrete_data = self.feature_enc.transform(discrete_data)
            if not self.discrete:
                X_df = pd.concat([pd.DataFrame(discrete_data.toarray()), continuous_data], axis=1)
            else:
                X_df = pd.DataFrame(discrete_data.toarray())
        else:
            X_df = continuous_data
        return X_df.values, y

    def inverse_transform(self, X):
        # Separate into the discrete and continuous components of the encoded data
        X_disc = X[:, :self.discrete_flen]
        X_cont = X[:, -self.continuous_flen:]

        # Reverse the One Hot Encoding of the discrete component
        X_df_disc = X[:, :self.discrete_flen]
        if self.discrete_flen > 0:
            X_df_disc = self.feature_enc.inverse_transform(X_disc)
        
        # Reverse the normalization of the continuous component
        X_df_cont = X[:, -self.continuous_flen:]
        if self.continuous_flen > 0:
            mean = np.array(self.mean)
            std = np.array(self.std)
            X_df_cont = X_cont * std + mean

        X_df = pd.DataFrame(np.concatenate((X_df_disc, X_df_cont), axis=1))

        #Convert continuous columns to 2 decimal places
        continuous_col = X_df.columns[-self.continuous_flen:]
        X_df[continuous_col] = X_df[continuous_col].apply(lambda x: pd.to_numeric(x).round(2))

        #Revert 
        X_df2 = X_df.copy()
        disc_idx = self.f_df.loc[self.f_df[1] == 'discrete'].index
        cont_idx = self.f_df.loc[self.f_df[1] == 'continuous'].index

        X_df.iloc[:, disc_idx] = X_df2.iloc[:, :-self.continuous_flen]
        X_df.iloc[:, cont_idx] = X_df2.iloc[:, -self.continuous_flen:]

        return X_df


def find_splits(db_enc):
    names = db_enc.X_fname[:db_enc.discrete_flen]
    temp = names[0][:2]
    idx = [0]
    for k in range(len(names)):
        if temp != names[k][:2]:
            idx.append(k)
            temp = names[k][:2]

    return idx


#sample_generator parameters:
#data = the actual dataset to be replicated
#db_enc = the DB Encoder fitted to the data
#size = number of data entried to be generated
#cont = "0" to generate continuous part from N(0,1), "1" to generate from Uniform(Quartile 1, Quartile 3), "2" to generate a fitted skewed normal distribution
#complete = "0" to generate both discrete and continuous parts, "1" to generate just the discrete part, "2" to generate just the continuous part
def sample_generator(data, db_enc, size, cont = 0, complete = 0, seed=None):
    np.random.seed(seed)
    #Split the dataset
    data_dc, data_c = data[:,:db_enc.discrete_flen], data[:,db_enc.discrete_flen:]
    gen_data = np.zeros((size, 1))
    
    #===================Generating the discrete part=======================#
    if (complete == 0 or complete == 1) & (db_enc.discrete_flen > 0):   
        idx = find_splits(db_enc)
        prob = np.sum(data_dc, axis = 0)/len(data_dc)
            
        splitted_prob = []
        for i in range(len(idx) - 1):
            splitted_prob.append(prob[idx[i] : idx[i+1]])
        splitted_prob.append(prob[idx[-1]:])
        
        for k in range(len(splitted_prob)):
            cum_prob = np.cumsum(splitted_prob[k])
            
            temp_data = np.zeros((size, len(cum_prob)))
            num = np.random.rand(size)
            for l in range(len(num)):
                for m in range(len(cum_prob)):
                    if num[l] < cum_prob[m]:
                        temp_data[l][m] = 1
                        break
                            
            gen_data = np.append(gen_data, temp_data, axis = 1)
    
    #===================Generating the continuous part======================#
    if (complete == 0 or complete == 2) & (db_enc.continuous_flen > 0):
        if cont == 0:
            cont_data = np.random.normal(0, 1, size = (size, data_c.shape[1]))
            gen_data = np.append(gen_data, cont_data, axis=1)

        elif cont == 1:
            q1 = np.quantile(data_c, 0.25, axis = 0)
            q3 = np.quantile(data_c, 0.75, axis = 0)

            for k in range(data_c.shape[1]):
                cont_data = np.random.uniform(q1[k], q3[k], size).reshape(size, 1)
                gen_data = np.append(gen_data, cont_data, axis = 1)

        elif cont == 2:
            for k in range(db_enc.continuous_flen):
                a, loc, scale = skewnorm.fit(data_c[:, k])
                cont_data = skewnorm(a, loc, scale).rvs(size).reshape(size, 1)
                gen_data = np.append(gen_data, cont_data, axis = 1)


    gen_data = np.delete(gen_data, 0, axis = 1)
    return gen_data


def class_balanced_few_shot_sample(X_train, y_train, n_samples, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # Convert y_train to Series if it's a single-column DataFrame
    if isinstance(y_train, pd.DataFrame):
        if y_train.shape[1] == 1:
            y_train = y_train.iloc[:, 0]
        else:
            raise ValueError("y_train must be a single-column DataFrame if passed as a DataFrame.")

    # Determine the number of unique classes in y_train
    if isinstance(y_train, pd.Series):
        unique_classes = y_train.unique()
    elif isinstance(y_train, np.ndarray):
        unique_classes = np.unique(y_train)
    else:
        raise TypeError("y_train must be a Pandas Series, single-column DataFrame, or NumPy array.")

    # Calculate the number of samples per class
    n_classes = len(unique_classes)
    n_samples_per_class = n_samples // n_classes

    # Check if n_samples is evenly divisible by the number of classes
    if n_samples % n_classes != 0:
        print(
            f"Warning: n_samples ({n_samples}) is not evenly divisible by the number of classes ({n_classes}). "
            f"Only {n_samples_per_class * n_classes} samples will be used."
        )

    # Initialize a list to hold the sampled indices
    sampled_indices = []

    # Sample data for each class
    for class_label in unique_classes:
        # Get the indices of the current class
        if isinstance(y_train, pd.Series):
            class_indices = y_train[y_train == class_label].index
        elif isinstance(y_train, np.ndarray):
            class_indices = np.where(y_train == class_label)[0]

        # Ensure there are enough samples for the class
        if len(class_indices) < n_samples_per_class:
            raise ValueError(
                f"Not enough samples for class {class_label}. "
                f"Requested {n_samples_per_class}, but only {len(class_indices)} are available."
            )

        # Randomly sample indices for the current class
        sampled_class_indices = np.random.choice(class_indices, n_samples_per_class, replace=False)
        sampled_indices.extend(sampled_class_indices)

    # Shuffle the combined indices to mix the class samples
    np.random.shuffle(sampled_indices)

    # Select the corresponding samples
    if isinstance(X_train, pd.DataFrame) and isinstance(y_train, (pd.Series, pd.DataFrame)):
        X_few_shot = X_train.loc[sampled_indices]
        y_few_shot = y_train.loc[sampled_indices]
    elif isinstance(X_train, np.ndarray) and isinstance(y_train, np.ndarray):
        X_few_shot = X_train[sampled_indices]
        y_few_shot = y_train[sampled_indices]
    else:
        raise TypeError("X_train and y_train must be either both NumPy arrays or both Pandas DataFrames/Series.")

    return X_few_shot, y_few_shot

def string_to_int(dict):
    for d in dict:
        if d['output'] == "1":
            d['output'] = 1
        else:
            d['output'] = 0
    return dict

def convert_to_binary(string):
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
    
def prompter(data):
    instruction=data['instruction']
    input=data['input']

    prompt=f"<s>[INST] <<SYS>>{instruction}<</SYS>> {input}[/INST]"

    return prompt

def extract_output(input):
    index_of_inst = input.find("[/INST]")
    return(input[index_of_inst + len("[/INST]"):].strip())

def TrainingSetGenerator(data):
    train_data = []
    for d in data:
        train_data.append({"messages":[
            {"role": "system", "content": d['instruction']},
            {"role": "user", "content": d['input']},
            {"role": "assistant", "content": d['output']}
        ]})
    return train_data

def calculate_classification_metrics(label, predicted):
    tp = sum(a == p == 1 for a, p in zip(label, predicted))
    tn = sum(a == p == 0 for a, p in zip(label, predicted))
    fp = sum(a == 0 and p == 1 for a, p in zip(label, predicted))
    fn = sum(a == 1 and p == 0 for a, p in zip(label, predicted))

    tpr = tp / (tp + fn) if tp + fn != 0 else 0
    tnr = tn / (tn + fp) if tn + fp != 0 else 0
    fpr = fp / (fp + tn) if fp + tn != 0 else 0
    fnr = fn / (fn + tp) if fn + tp != 0 else 0

    return tpr, tnr, fpr, fnr

def class_balanced_serialized_sample(data, n_shots, label_key='output', seed=None):
    """
    Perform class balanced sampling on the serialized data json.
    
    Parameters:
      data (list): List of dictionaries, each representing an instance.
      n_shots (int): Total number of samples desired.
      label_key (str): The key used for labels in each dictionary.
      seed (int, optional): Random seed for reproducibility.
    
    Returns:
      list: A list of sampled dictionaries, balanced across the unique classes.
    """
    if seed is not None:
        random.seed(seed)

    # Get unique classes from the data for the specified label_key
    unique_classes = list({d[label_key] for d in data})
    num_classes = len(unique_classes)
    
    # Desired samples per class (using floor division)
    shots_per_class = n_shots // num_classes
    sampled_by_class = {}
    
    # Sample for each class
    for cls in unique_classes:
        class_samples = [d for d in data if d[label_key] == cls]
        if len(class_samples) < shots_per_class:
            # Warn if there are not enough samples and take all available
            print(f"Warning: Not enough samples for class {cls}. "
                  f"Requested {shots_per_class}, but only found {len(class_samples)}.")
            sampled_by_class[cls] = class_samples
        else:
            sampled_by_class[cls] = random.sample(class_samples, shots_per_class)
    
    # Combine sampled instances
    sampled_data = []
    for cls in unique_classes:
        sampled_data.extend(sampled_by_class[cls])
    
    # If total samples are less than desired (due to shortage in one or more classes),
    # add extra samples from those classes that have extras.
    total_samples = len(sampled_data)
    if total_samples < n_shots:
        needed = n_shots - total_samples
        extras = []
        for cls in unique_classes:
            all_class_samples = [d for d in data if d[label_key] == cls]
            current_samples = sampled_by_class[cls]
            # Find remaining instances (preserving order is not crucial here)
            remaining = [d for d in all_class_samples if d not in current_samples]
            extras.extend(remaining)
        if len(extras) >= needed:
            sampled_data.extend(random.sample(extras, needed))
        else:
            sampled_data.extend(extras)
    
    # Shuffle final sampled data
    random.shuffle(sampled_data)
    return sampled_data
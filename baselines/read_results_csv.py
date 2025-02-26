import pandas as pd
import os


def load_results_baselines(dataset, save_dir=None):

    if save_dir is None:
        save_dir = f'results_{dataset}'
    with open(os.path.join(save_dir, f'baseline_results_{dataset}.txt'), 'r') as resfile:

        lines = resfile.readlines()

    all_metrics = ['AUC', 'Acc', 'F1', 'TPR', 'FPR', 'rules', 'conditions']
    models = {}
    model_flag = False
    current_model = None
    for line in lines:

        if line.startswith('---'):
            model_flag = True
            continue
        elif line.startswith('\n'):
            continue
        elif line.startswith('No'):
            metric = line.split(' ')[1]
            try:
                models[current_model].pop(metric)
            except:
                print(metric, line)
            continue
        elif line.startswith('AUCS'):
            break
        if model_flag:
            line = line[:-1]
            models[line] = {key:{'Mean': 0, 'STD': 0} for key in all_metrics}
            current_model = line
            model_flag = False
        else:
            statistic, metric, score = line.split(' ')
            metric = metric[:-1]
            models[current_model][metric][statistic] = round(float(score),2)

    return models

def pivot_metrics(models):
    # Flatten the nested dictionary into a list of dictionaries
    flattened_data = []
    for method, results in models.items():
        for metric, values in results.items():
            flattened_data.append(
                {'Method': method, 'Metric': metric, 'Score': f"{values['Mean']} +/- {values['STD']}"})

    # Create a DataFrame from the flattened data
    df = pd.DataFrame(flattened_data)

    # Pivot the DataFrame to have AUC, Acc, and F1 as columns
    df = df.pivot(index='Method', columns='Metric', values='Score')

    # Reset the index for a cleaner DataFrame
    df.reset_index(inplace=True)

    # Rename the columns for clarity
    df.columns.name = None  # Remove the column name
    # df = df.rename(columns={'AUC': 'Mean_AUC', 'Acc': 'Mean_Acc', 'F1': 'Mean_F1'})

    return df

def save_as_excel(df, dataset, save_dir=None, bold=False):
    # print(save_dir)
    if save_dir is None:
        save_dir = f'results_{dataset}'
    # Define the filename for the Excel file
    excel_filename = os.path.join(save_dir, 'results.xlsx')

    # Save the DataFrame to an Excel file
    with pd.ExcelWriter(excel_filename, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Sheet1', index=False)

        if bold:
            # Get the xlsxwriter workbook and worksheet objects
            workbook = writer.book
            worksheet = writer.sheets['Sheet1']

            # Create a format to apply bold style
            bold_format = workbook.add_format({'bold': True})

            # Iterate through columns and find maximum values
            for col_num, column in enumerate(df.columns[1:], 1):
                # print(column)
                # print(df[column])
                # print(df[column].str)#.split(' +/-'))

                # Define a regular expression pattern to extract the mean and standard deviation
                pattern = r'(\d+\.\d+) \+/- (\d+\.\d+)'

                # Extract the mean and standard deviation using the regular expression
                df['mean'] = df[column].str.extract(pattern, expand=True)[0].astype(float)
                df['std'] = df[column].str.extract(pattern, expand=True)[1].astype(float)

                # Calculate the maximum value by adding the mean and standard deviation
                df['max_value'] = df['mean'] + df['std']

                # Find the row with the maximum value
                if column not in ['rules', 'conditions']:
                    row_with_max_value = df[df['max_value'] == df['max_value'].max()]
                else:
                    row_with_max_value = df[df['max_value'] == df['max_value'].min()]
                max_row = row_with_max_value.index[0]
                max_value = df[column].iloc[max_row]#row_with_max_value['max_value'].max()

                # max_value = df[column].str.split(' +/-').str[0].astype(float).max()
                #
                # # Find the row number of the maximum value
                # max_row = df[df[column].str.split(' +/-').str[0].astype(float) == max_value].index[0] + 2  # Add 2 to account for 0-based indexing and header row

                # Apply bold style to the maximum value
                worksheet.write(max_row, col_num, max_value, bold_format)

    # Save the Excel file with bold maximum values
    print(f"Excel file saved as '{excel_filename}'")


def merge_excels(datasets, save_dir=None, output_excel_file='merged_excel.xlsx'):

    if save_dir is None:
        save_dir = [f'results_{dataset}' for dataset in datasets]
    # List of the Excel files you want to merge
    excel_files = [os.path.join(s, 'results.xlsx') for s in save_dir]  # Replace with your actual file paths



    map_sheet = {f'Sheet{i+1}': file.split('.')[0].split('results_')[0][:-1] for i,file in enumerate(excel_files)}
    with pd.ExcelWriter(output_excel_file, engine='xlsxwriter') as writer:
        # Iterate over the list of Excel files and write each to a separate sheet
        for i, file in enumerate(excel_files):
            # Extract the sheet name from the filename (without the extension)
            sheet_name = file.split('/')[-2]
            # print(sheet_name.split('results_'))
            # print(file, sheet_name)
            # sheet_name = sheet_name.split('results')
            # print(sheet_name)
            # if sheet_name[1][:-1] != '':
            #     sheet_name = sheet_name[1][:-1]
            # else:
            #     sheet_name = sheet_name[0]
            #     sheet_name = sheet_name.split('/')[1]

            # Read the Excel file into a DataFrame
            df = pd.read_excel(file)

            # Write the DataFrame to the Excel writer with the sheet name
            df.to_excel(writer, sheet_name=sheet_name, index=False)


    # print(map_sheet)
    # sheet_names = list(writer.sheets.keys())
    # workbook = load_workbook(output_excel_file)
    # for i, sheet in enumerate(sheet_names):
    #     worksheet = workbook.get_sheet_by_name(sheet)
    #     worksheet.title = map_sheet[sheet]
    # workbook.save(output_excel_file)
    # Save the merged Excel file
    # writer.close()


if __name__ == '__main__':

    datasets = ['cots_6M_red', 'cots_12M_red', 'cots_24M_red', 'cots_6M', 'cots_12M', 'cots_24M']
    for dataset in datasets:
        models = load_results_baselines(dataset)
        df = pivot_metrics(models)

        save_as_excel(df, dataset)

    # datasets = ['cots_6M', 'cots_12M', 'cots_24M']
    # for dataset in datasets:
    #     df = load_results_baselines(dataset)
    #     save_as_excel(df, dataset)

    merge_excels(datasets)

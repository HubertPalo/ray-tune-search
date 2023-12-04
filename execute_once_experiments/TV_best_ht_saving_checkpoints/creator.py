import os
import yaml
from tqdm import tqdm
from copy import deepcopy
import pandas as pd
from execute_once_experiments.one_exec_helper import Template

datasets = ['kuhar', 'motionsense', 'uci', 'wisdm', 'realworld_thigh', 'realworld_waist']
percentages = [25]
models = ['ae', 'tae', 'convae', 'convtae']
# models = ['ae']

    
for model in tqdm(models, desc='Models'):
    template_file = f'TVT_template_{model}.yaml'
    if model in ['ae', 'tae', 'convae', 'convtae']:
        template_file = f'TVT_template_convtae.yaml'
    # Read the basic file
    with open(template_file) as file:
        original_template = yaml.load(file, Loader=yaml.FullLoader)
    template_obj = Template(original_template)
    # Create the folders
    os.makedirs(f'../TVT_sb_{model}', exist_ok=True)
    os.makedirs(f'../TVT_sb_{model}/configs', exist_ok=True)
    os.makedirs(f'../TVT_sb_{model}/results', exist_ok=True)
    os.makedirs(f'../TVT_sb_{model}/scores', exist_ok=True)
    for dataset in tqdm(datasets, desc='Datasets'):
        template_obj.update_dataset(dataset)
        for percentage in percentages:
            # Get the data
            data_file = f'../../experiments/{model}_{dataset}_P{percentage}/data.csv'
            if model in ['ae', 'tae', 'convae', 'convtae']:
                data_file = f'../../experiments/P10_{model}_{dataset}_P{percentage}/data.csv'
            try:
                data = pd.read_csv(data_file)
                config_columns = [col for col in data.columns if 'config' in col]
                config_for_best_score = data[config_columns].iloc[data['score'].idxmax(), :]
                template_updated = template_obj.update(model, config_for_best_score)
                # print(template_updated)
                # Create the file
                with open(f'../TVT_sb_{model}/configs/TVT_sb_{model}_{dataset}_P{percentage}.yaml', 'w') as file:
                    yaml.dump(template_updated, file)
            except Exception as e:
                print(e)
                print(f'No file for {model} {dataset} {percentage}', data_file)

            # best_config_file = f'../../experiments/{model}_{dataset}_P{percentage}/best.yaml'
            # if model in ['ae', 'tae', 'convae', 'convtae']:
            #     best_config_file = f'../../experiments/P10_{model}_{dataset}_P{percentage}/best.yaml'
            # try:
            #     with open(best_config_file) as file:
            #         best_config = yaml.load(file, Loader=yaml.FullLoader)
            #         # print(best_config)
            #     # Update the template
            #     template_updated = template_obj.update(model, best_config['config'])
                
            # except:
            #     print(f'No file for {model} {dataset} {percentage}', best_config_file)
            # Create the file
            # with open(f'TVT_sb_{model}/configs/TVT_sb_{model}_{dataset}_P{percentage}.yaml', 'w') as file:
            #     yaml.dump(TVT_template, file)
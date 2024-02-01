import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import yaml
from tqdm import tqdm
from copy import deepcopy
import pandas as pd
from basic.template import Template

datasets = ['kuhar', 'motionsense', 'uci', 'wisdm', 'realworld_thigh', 'realworld_waist']
percentages = [2.5, 5, 25, 50, 75, 100, 200]
models = ['ae', 'tae', 'convae', 'convtae', 'umap']
# Create folder structure
os.makedirs(f'../TVT_sb_best_found_2024', exist_ok=True)
os.makedirs(f'../TVT_sb_best_found_2024/configs', exist_ok=True)
os.makedirs(f'../TVT_sb_best_found_2024/results', exist_ok=True)
os.makedirs(f'../TVT_sb_best_found_2024/scores', exist_ok=True)

for model in tqdm(models, desc='Models'):
    template_file = f'TVT_template_{model}.yaml'
    if model in ['ae', 'tae', 'convae', 'convtae']:
        template_file = f'TVT_template_convae.yaml'
    # Read the basic file
    with open(template_file) as file:
        original_template = yaml.load(file, Loader=yaml.FullLoader)
    # Create the folders
    for dataset in tqdm(datasets, desc=model):
        template_obj = Template(deepcopy(original_template))
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
                # Create the file
                with open(f'../TVT_sb_best_found_2024/configs/TVT_sb_{model}_{dataset}_P{percentage}.yaml', 'w') as file:
                    yaml.dump(template_updated, file)
            except Exception as e:
                print(e)
                print(f'No file for {model} {dataset} {percentage}', data_file)
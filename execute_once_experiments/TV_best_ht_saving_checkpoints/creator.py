import os
import yaml
from tqdm import tqdm
from copy import deepcopy
import pandas as pd


class Template:
    def __init__(self, base_template):
        self.base_template = base_template
    
    def update_with_umap_config(self, config):
        template = deepcopy(self.base_template)
        template['reducer']['kwargs']['n_components'] = int(config['config/umap_ncomp'])
        template['reducer']['kwargs']['n_epochs'] = int(config['config/umap_epochs'])
        template['reducer']['kwargs']['n_neighbors'] = int(config['config/umap_neigh'])
        template['reducer']['kwargs']['min_dist'] = float(config['config/umap_mdist'])
        template['reducer']['kwargs']['spread'] = float(config['config/umap_spread'])
        return template
    
    def update_with_ae_config(self, config):
        # print(config)
        # print('START UPDATE')
        # print(self.base_template)
        template = deepcopy(self.base_template)
        # Autoencoder part
        template['reducer']['kwargs']['batch_size'] = int(config['config/batch_size'])
        template['reducer']['kwargs']['latent_dim'] = int(config['config/latent_dim'])
        template['reducer']['kwargs']['extra_properties']['num_HL'] = int(config['config/num_HL'])
        template['reducer']['kwargs']['extra_properties']['optimizer_lr'] = float(config['config/opt_lr'])
        template['reducer']['kwargs']['extra_properties']['optimizer_weight_decay'] = float(config['config/opt_wd'])
        # print('FIN UPDATE', template)
        return template
    
    def update_with_tae_config(self, config):
        template = deepcopy(self.base_template)
        # Autoencoder part
        template['reducer']['kwargs']['batch_size'] = int(config['config/batch_size'])
        template['reducer']['kwargs']['latent_dim'] = int(config['config/latent_dim'])
        template['reducer']['kwargs']['extra_properties']['num_HL'] = int(config['config/num_HL'])
        template['reducer']['kwargs']['extra_properties']['optimizer_lr'] = float(config['config/opt_lr'])
        template['reducer']['kwargs']['extra_properties']['optimizer_weight_decay'] = float(config['config/opt_wd'])
        # Topological part
        template['reducer']['kwargs']['model_lambda'] = float(config['config/m_lambda'])
        return template
    
    def update_with_convae_config(self, config):
        template = deepcopy(self.base_template)
        # Autoencoder part
        template['reducer']['kwargs']['batch_size'] = int(config['config/batch_size'])
        template['reducer']['kwargs']['latent_dim'] = int(config['config/latent_dim'])
        template['reducer']['kwargs']['extra_properties']['num_HL'] = int(config['config/num_HL'])
        template['reducer']['kwargs']['extra_properties']['optimizer_lr'] = float(config['config/opt_lr'])
        template['reducer']['kwargs']['extra_properties']['optimizer_weight_decay'] = float(config['config/opt_wd'])
        # Convolutional part
        template['reducer']['kwargs']['extra_properties']['num_CL'] = int(config['config/num_CL'])
        template['reducer']['kwargs']['extra_properties']['size_CL'] = int(config['config/size_CL'])
        template['reducer']['kwargs']['extra_properties']['kernel_size'] = int(config['config/kernel'])
        return template
    
    def update_with_convtae_config(self, config):
        template = deepcopy(self.base_template)
        # Autoencoder part
        template['reducer']['kwargs']['batch_size'] = int(config['config/batch_size'])
        template['reducer']['kwargs']['latent_dim'] = int(config['config/latent_dim'])
        template['reducer']['kwargs']['extra_properties']['num_HL'] = int(config['config/num_HL'])
        template['reducer']['kwargs']['extra_properties']['optimizer_lr'] = float(config['config/opt_lr'])
        template['reducer']['kwargs']['extra_properties']['optimizer_weight_decay'] = float(config['config/opt_wd'])
        # Topological part
        template['reducer']['kwargs']['model_lambda'] = float(config['config/m_lambda'])
        # Convolutional part
        template['reducer']['kwargs']['extra_properties']['num_CL'] = int(config['config/num_CL'])
        template['reducer']['kwargs']['extra_properties']['size_CL'] = int(config['config/size_CL'])
        template['reducer']['kwargs']['extra_properties']['kernel_size'] = int(config['config/kernel'])
        return template

    
    def update_dataset(self, dataset):
        for key, item in enumerate(self.base_template['reducer_dataset']):
            self.base_template['reducer_dataset'][key] = item.replace('dataset_to_replace', dataset)
        for key, item in enumerate(self.base_template['train_dataset']):
            self.base_template['train_dataset'][key] = item.replace('dataset_to_replace', dataset)
        for key, item in enumerate(self.base_template['test_dataset']):
            self.base_template['test_dataset'][key] = item.replace('dataset_to_replace', dataset)

    def update(self, model, config):
        func_name = f'update_with_{model}_config'
        if hasattr(self, func_name) and callable(func := getattr(self, func_name)):
            return func(config)


# Objective: create TV configs for the best of every combination (dataset, model and 25 percentage)
datasets = ['kuhar', 'motionsense', 'uci', 'wisdm', 'realworld_thigh', 'realworld_waist']
percentages = [25]
models = ['ae', 'tae', 'convae', 'convtae']


for model in tqdm(models, desc='Models'):
    template_file = f'TV_template_{model}.yaml'
    if model in ['ae', 'tae', 'convae', 'convtae']:
        template_file = f'TV_template_convae.yaml'
    template_file = '../' + template_file
    # Read the template file
    with open(template_file) as file:
        original_template = yaml.load(file, Loader=yaml.FullLoader)
    
    # Create the config folder
    os.makedirs(f'configs', exist_ok=True)
    os.makedirs(f'results', exist_ok=True)
    os.makedirs(f'files', exist_ok=True)
    os.makedirs(f'scores', exist_ok=True)
    # TV_best_found_{model}_{dataset}_{percentage}
    for dataset in tqdm(datasets, desc='Datasets'):
        template_obj = Template(deepcopy(original_template))
        template_obj.update_dataset(dataset)
        for percentage in percentages:
            # Get the data
            data_file = f'{model}_{dataset}_P{percentage}/data.csv'
            if model in ['ae', 'tae', 'convae', 'convtae']:
                data_file = 'P10_' + data_file
            data_file = '../../experiments/' + data_file
            try:
                data = pd.read_csv(data_file)
                config_columns = [col for col in data.columns if 'config' in col]
                config_for_best_score = data[config_columns].iloc[data['score'].idxmax(), :]
                template_updated = template_obj.update(model, config_for_best_score)
                template_updated['reducer']['kwargs']['save_frequency'] = 'best'
                template_updated['reducer']['kwargs']['save_tag'] = f'TV_sb_gradual_{model}_{dataset}_{percentage}'
                template_updated['reducer']['kwargs']['save_dir'] = f'execute_once_experiments/TV_best_ht_saving_checkpoints/files/'
                # print(template_updated)
                # Create the file
                with open(f'configs/TV_sb_{model}_{dataset}_P{percentage}.yaml', 'w') as file:
                    yaml.dump(template_updated, file)
            except Exception as e:
                print(e)
                print(f'No file for {model} {dataset} {percentage}', data_file)
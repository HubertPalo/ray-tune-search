import os
import yaml
from tqdm import tqdm
from copy import deepcopy
import pandas as pd


datasets = ['kuhar', 'motionsense', 'uci', 'wisdm', 'realworld_thigh', 'realworld_waist']
percentages = [25, 50, 75, 100, 200]
models = ['ae', 'tae', 'convae', 'convtae']
# models = ['ae']
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
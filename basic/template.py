from copy import deepcopy

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
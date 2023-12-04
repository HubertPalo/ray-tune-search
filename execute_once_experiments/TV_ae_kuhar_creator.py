import numpy as np
import os
import yaml
import pickle


prefix = 'tae_kuhar'
os.makedirs(f'execute_once_experiments/TV_sb_gradual_{prefix}', exist_ok=True)
os.makedirs(f'execute_once_experiments/TV_sb_gradual_{prefix}/configs', exist_ok=True)
os.makedirs(f'execute_once_experiments/TV_sb_gradual_{prefix}/results', exist_ok=True)
os.makedirs(f'execute_once_experiments/TV_sb_gradual_{prefix}/scores', exist_ok=True)
with open(f'execute_once_experiments/TV_best_ht_saving_checkpoints/configs/TV_sb_{prefix}_P25.yaml') as file:
    original_template = yaml.load(file, Loader=yaml.FullLoader)

for file in os.listdir():
    if file.startswith(prefix+'_epoch'):
        original_template['reducer']['kwargs']['file_to_load'] = file
        with open(f'execute_once_experiments/TV_sb_gradual_{prefix}/configs/TV_sb_gradual_{file[:-4]}.yaml', 'w') as file:
            yaml.dump(original_template, file)
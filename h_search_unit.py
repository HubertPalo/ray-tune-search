# Python imports
import os
from pathlib import Path
import sys
import yaml
import numpy as np
# Add upper directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Third-party imports
from basic.config import *
# from basic.run_basic_experiment import run_basic_experiment
# from basic.run_custom_experiment import run_custom_experiment
from basic.run_experiment import run_experiment
from basic.helper import process_result_default, process_result_custom


def get_score(max_report, extra_info=None, function='mean'):
    if 'max' in max_report:
        return max_report['max']
    accuracies = [max_report[dataset]/extra_info[dataset] for dataset in max_report]
    functions_to_apply = {
        'mean': np.mean,
        'min': np.min,
        'max': np.max
    }
    return functions_to_apply[function](accuracies)

def h_search_unit(
        dataset_locations,
        save_folder=None,
        config_to_execute:ExecutionConfig=None,
        specific_name=None,
        # experiment_type='default',
        additional_info={}
        ):
    # Set the random state
    # set_random_state(random_state)
    # Create the experiment config
    # experiment_config = umap_simple_experiment(config, dataset, random_state)
    # experiment_functions = {
    #     'default': run_basic_experiment,
    #     'custom': run_custom_experiment
    # }
    baselines = additional_info['BASELINES'] if 'BASELINES' in additional_info else {}
    baseline_gain = additional_info['BASELINE-GAIN'] if 'BASELINE-GAIN' in additional_info else 'none'
    process_functions = {
        'default': process_result_default,
        'tests_isolated': process_result_custom
    }
    
    # Get the experiment function
    # experiment_function = experiment_functions[experiment_type]
    # Get the process function
    
    process_function = process_functions[config_to_execute.metadata.experiment_type]
    # Run the experiment
    experiment_result = run_experiment(
        dataset_locations=dataset_locations,
        config_to_execute=config_to_execute
    )
    
    # Save the results
    if save_folder:
        # Get the number of files in the folder
        item = len(os.listdir(save_folder))
        if specific_name:
            item = specific_name
        # Save the results
        with open(f"{save_folder}/{item}.yaml", "w") as f:
            yaml.dump(experiment_result, f)
    # Defining the result object
    result_object = dict()
    # Processing the report
    max_report, report = process_function(experiment_result['report'])
    # General Report
    result_object.update(report)
    # Topology Report
    topology_result = experiment_result['additional']['pydrm_report'] if 'pydrm_report' in experiment_result['additional'] else {}
    result_object.update(topology_result)
    # Reducer size Report
    reducer_size = {
        'num_params': experiment_result['additional']['num_params'] if 'num_params' in experiment_result['additional'] else 0,
        'num_trainable_params': experiment_result['additional']['num_trainable_params'] if 'num_trainable_params' in experiment_result['additional'] else 0
    }
    result_object.update(reducer_size)
    # if 'num_params' in experiment_result['additional']:
    #     result_object['num_params'] = experiment_result['additional']['num_params']
    # if 'num_trainable_params' in experiment_result['additional']:
    #     result_object['num_trainable_params'] = experiment_result['additional']['num_trainable_params']
    # Score Report
    result_object['score'] = get_score(max_report, extra_info=baselines, function=baseline_gain)
    # max_accuracy = processed_result[-1]['accuracy']
    # result_object = {'score': max_accuracy}
    # criteria = ['accuracy (mean)', 'accuracy (std)', 'f1-score macro (mean)', 'f1-score macro (std)', 'f1-score weighted (mean)', 'f1-score weighted (std)']
    # for result in processed_result[:-1]:
    #     for criterion in criteria:
    #         if criterion in result:
    #             result_object[f'{result["estimator"]}-{criterion}'] = result[criterion]

    # score = process_result(experiment_result)[-1]['accuracy']
    # if topology_result:
    #     result_object.update(topology_result)
    # return {'score': max_accuracy, 'num_params': num_params, 'num_trainable_params': num_trainable_params}
    return result_object
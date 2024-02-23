from ray.air import session
from copy import deepcopy
import sys
import traceback
from dacite import from_dict

from basic.config import ExecutionConfig
from h_search_unit import h_search_unit


def default_objective_function(
    config,
    save_folder,
    dataset_locations,
    basic_experiment_configuration=None,
    search_space=None):
    basic_experiment_config = deepcopy(basic_experiment_configuration)
    # Update the values for the current experiment
    for key in search_space:
        property_content = config[key] if key in config else []
        if search_space[key]['tune_function'] == 'multichoice':
            # Get the values
            property_content = []
            for parameter in search_space[key]['tune_parameters']:
                multichoice_key = f"MC-{key}-{parameter}"
                if config[multichoice_key] == 1:
                    property_content.append(parameter) 
        # Prepare the route
        route = search_space[key]['route'].split('/')
        property_to_modify = basic_experiment_config
        for key, item in enumerate(route[:-1]):
            # If item is a number, then it is a list
            if item.isdigit():
                item = int(item)
            property_to_modify = property_to_modify[item]
        # Set the value
        property_to_modify[route[-1]] = property_content
    config_to_execute = from_dict(
        data_class=ExecutionConfig,
        data=basic_experiment_config
    )
    try:
        result = h_search_unit(
            save_folder=save_folder,
            dataset_locations=dataset_locations,
            config_to_execute=config_to_execute
        )
    except Exception as e:
        print('EXCEPTION FOUND\n', e)
        syserror = sys.exc_info()
        result = {
            'score': -0.001,
            'num_params': -1,
            'num_trainable_params': -1,
            'error_type': str(syserror[0]),
            'error_message': str(syserror[1]),
            'error_traceback': '\n'.join(traceback.format_tb(e.__traceback__))
        }
    session.report(result)
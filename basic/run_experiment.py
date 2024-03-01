from typing import Dict
from librep.config.type_definitions import PathLike
from basic.config import ExecutionConfig
from basic.run_basic_experiment import run_basic_experiment
from basic.run_custom_experiment import run_custom_experiment

# Function that runs the experiment
def run_experiment(
    dataset_locations: Dict[str, PathLike],
    config_to_execute: ExecutionConfig
) -> dict:
    functions = {
        'default': run_basic_experiment,
        'tests_isolated': run_custom_experiment
    }
    return functions[config_to_execute.metadata.experiment_type](
        dataset_locations=dataset_locations,
        config_to_execute=config_to_execute
    )
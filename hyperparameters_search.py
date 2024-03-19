from ray import tune, air
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search import ConcurrencyLimiter
import yaml
import os
from basic.helper import set_random_state
from ray.tune import Callback
from ray.tune.stopper import Stopper
import pandas as pd
from pathlib import Path
import functools
from hs_objective_function import default_objective_function, new_objective_function
from basic.exploration_config import ExplorationConfig


class BestResultCallback(Callback):

    def __init__(self, experiment_full_path):
        self.experiment_full_path = experiment_full_path
        self.data = []
        self.errors = []
        self.counter = 0

    def on_trial_result(self, iteration, trials, trial, result, **info):
        new_row = {
            'iteration': iteration,
            'trial_id': trial.trial_id,
            'score': result['score'],
            'config': result['config']
        }
        self.data.append(new_row)
        self.counter += 1
        if self.counter % 50 == 0:
            data_df = pd.DataFrame(self.data)
            data_df.to_csv(f"{self.experiment_full_path}/callback_data.csv", index=False)
        if 'error_message' in result:
            self.errors.append(
                {
                    'trial_id': trial.trial_id,
                    'config': result['config'],
                    'error_type': result['error_type'],
                    'error_message': result['error_message'],
                    'error_traceback': result['error_traceback']
                }
            )
            errors_df = pd.DataFrame(self.errors)
            errors_df.to_csv(f"{self.experiment_full_path}/callback_errors.csv", index=False)

class CustomStopper(Stopper):
    def __init__(
        self,
        metric: str,
        min: int = 1000,
        patience: int = 100,
        experiment_full_path = ''
    ):
        self._metric = metric
        self._patience = patience
        self._iterations = 0
        self._min = min
        self.best_found = 0
        self.counter = 0
        self.trial_ids = []
        self.experiment_full_path = experiment_full_path

    def __call__(self, trial_id, result):
        print(f"CUSTOM STOPPER - Trial ids length: {len(self.trial_ids)} counter: {self.counter}")
        # if self.experiment_full_path != '':
            # print(f"CUSTOM STOPPER - Saving results to {self.experiment_full_path}...")
        if result[self._metric] is not None and trial_id not in self.trial_ids and result[self._metric] > 0:
            self.trial_ids.append(trial_id)
            self._iterations += 1
            # self.results.append(result[self._metric])
            self.counter += 1
            if result[self._metric] > self.best_found:
                self.best_found = result[self._metric]
                self.counter = 0
        # print(f"Iterations: {self._iterations}")
        # print(f"Counter: {self.counter}")
        # print(f'Results: {self.results}')
        return self.stop_all() #  or self.too_much_errors()
    
    def too_much_errors(self):
        errors_path = self.experiment_full_path / 'callback_errors.csv'
        if os.path.exists(errors_path):
            try:
                errors_df = pd.read_csv(errors_path)
                if len(errors_df) > 10:
                    print('TOO MUCH ERRORS - Stopping the experiment...')
                    return True
                return False
            except pd.errors.EmptyDataError:
                return False
        return False
        
    def stop_all(self):
        return self._iterations > self._min and self.counter > self._patience


def hyperparameters_search(
        dataset_locations=None,
        base_config=None,
        exploration_config:ExplorationConfig=None,
        experiment_full_path=None,
        experiment_info=None):
    # Set the random state
    if experiment_info['random_state'] != -1:
        set_random_state(experiment_info['random_state'])
    
    # Get the search space, initial params and experiment name from the config file
    search_space = {}
    # multichoice_info = []
    # for key, value in exploration_config.search_space.items():
    for search_space_unit in exploration_config.search_space:
        # if value['tune_function'] == 'multichoice':
        if search_space_unit.tune_function == 'multichoice':
            # multichoice_info.append((key, value))
            keys = [f"MC-{search_space_unit.identifier}-{i}" for i in search_space_unit.tune_parameters]
            for k in keys:
                search_space[k] = tune.choice([0,1])
            continue
        # elif search_space_unit.tune_function == 'choice':
        #     search_space[search_space_unit.identifier] = tune.choice(*[search_space_unit.tune_parameters])
        else:
            search_space[search_space_unit.identifier] = getattr(tune, search_space_unit.tune_function)(*search_space_unit.tune_parameters)
    # search_space = {
    #     key: getattr(tune, value['tune_function'])(*value['tune_parameters'])
    #     for key, value in exploration_config["search_space"].items()
    # }
    initial_params = exploration_config.initial_params
    
    resources = {'cpu':exploration_config.resources.cpu, 'gpu':exploration_config.resources.gpu}
    
    # Create the experiments folder
    save_folder = None
    if experiment_info['save_experiment']:
        save_folder = os.path.abspath(f'{experiment_full_path}/files')
        print(f"Saving results to {save_folder}...")
        os.makedirs(save_folder, exist_ok=True)
    
    # Create the callback errors file
    callback_errors_csv = os.path.abspath(f'{experiment_full_path}/callback_errors.csv')
    print(f"Saving errors in file {callback_errors_csv}...")
    pd.DataFrame().to_csv(callback_errors_csv, index=False)

    hyperopt = HyperOptSearch(points_to_evaluate=initial_params)
    hyperopt = ConcurrencyLimiter(hyperopt, max_concurrent=experiment_info['max_concurrent'])

    # Initializing the trainable
    # trainables = {
    #     'default': default_objective_function,
    #     'new': new_objective_function
    # }
    # trainable = trainables[experiment_info['objective_function']]
    trainable = default_objective_function

    baseline_gain = experiment_info['baseline_gain']
    additional_info = {'BASELINE-GAIN': baseline_gain}
    print(f"Baseline gain: {baseline_gain}")
    print(f"Additional info: {exploration_config.additional_info}")
    
    if exploration_config.additional_info:
        additional_info.update(exploration_config.additional_info)

    
    # Setting the parameters for the function
    trainable = tune.with_parameters(
        trainable,
        save_folder=save_folder,
        dataset_locations=dataset_locations,
        basic_experiment_configuration=base_config,
        exploration_configuration=exploration_config,
        additional_info=additional_info
    )
    # Allocating the resources needed
    trainable = tune.with_resources(trainable=trainable, resources=resources)
    tuner = tune.Tuner(
        trainable=trainable,
        tune_config=tune.TuneConfig(
            metric="score",
            mode="max",
            num_samples=-1,
            scheduler=ASHAScheduler(),
            search_alg=hyperopt,
            time_budget_s=experiment_info['time_budget'],
        ),
        run_config=air.RunConfig(
            name=str(experiment_full_path).split('/')[-1],
            callbacks=[BestResultCallback(experiment_full_path)],
            stop=CustomStopper(metric="score", min=1000, patience=100, experiment_full_path=experiment_full_path)
            # stop=ExperimentPlateauStopper(metric="score", std=0.001, top=10, mode="max", patience=0)
        ),
        param_space=search_space
    )
    if experiment_info['restore']:
        print('Restoring the hyperparameters search...')
        print("TUNE_ORIG_WORKING_DIR:", os.environ.get("TUNE_ORIG_WORKING_DIR"))
        print("TUNE_WORKING_DIR:", os.environ.get("TUNE_WORKING_DIR"))
        print("TUNE_RESULT_DIR:", os.environ.get("TUNE_RESULT_DIR"))
        # Resume experiment with: Tuner.restore(path="/umap_kuhar_p10_2", trainable=...)
        restore_path = f'/home/darlinne.soto/ray_results/{str(experiment_full_path).split("/")[-1]}'
        restore_path = Path(restore_path).as_posix()
        print(restore_path)
        fixed_func = functools.partial(tune.Tuner.restore, path=restore_path, trainable=trainable)
        tuner = fixed_func()
    print('Starting the hyperparameters search...')
    results = tuner.fit()
    print('Finished the hyperparameters search...')
    # Save results in a csv file
    results.get_dataframe().to_csv(f"{experiment_full_path}/data.csv")
    # Report the best result
    best_result = results.get_best_result(metric="score", mode="max")
    to_save = {'config': best_result.config, 'score': float(best_result.metrics['score'])}

    # Save the best result
    with open(f"{experiment_full_path}/best.yaml", "w") as f:
        yaml.dump(to_save, f)
from hyperparameters_search import hyperparameters_search
from ray import tune
import argparse
from basic.helper import get_dataset_locations
from pathlib import Path
import yaml
from dacite import from_dict
from basic.config import ExecutionConfig
import os


# Main function
def main(args):
    # Get the dataset locations
    data_fullpath = Path.absolute(Path(args.data))
    if not os.path.exists(data_fullpath):
        raise ValueError(f"Data path {data_fullpath} does not exist")
    dataset_locations_fullpath = Path.absolute(Path(args.dataset_locations_fullpath))
    if not os.path.exists(dataset_locations_fullpath):
        raise ValueError(f"Dataset locations path {dataset_locations_fullpath} does not exist")
    dataset_locations = get_dataset_locations(
        data_fullpath=data_fullpath,
        dataset_locations_fullpath=dataset_locations_fullpath
    )

    folders_to_check = [
        Path(f"{args.experiment}"),
        Path(f"experiments/{args.experiment}")
    ]

    for folder in folders_to_check:
        print(f"Looking for files inside path {folder}...")
        exploration_config_path = Path(f"{folder}/exploration_config.yaml")
        base_config_path = Path(f"{folder}/base_config.yaml")
        try:
            # Read the hyperparameters search config file
            with open(exploration_config_path, "r") as f:
                exploration_config = yaml.load(f, Loader=yaml.FullLoader)
            with open(base_config_path, "r") as f:
                base_config = yaml.load(f, Loader=yaml.FullLoader)
            print(f"Files inside path {folder} found.")
            experiment_full_path = Path.absolute(Path(f"{folder}"))
            break
        except:
            exploration_config = None
            base_config = None
            print(f"Files inside path {args.experiment} not found...")

    if exploration_config is None or base_config is None:
        raise ValueError(f"No experiment files found. Exiting...")
    
    time_budget = args.time_budget
    if time_budget == -1:
        time_budget = None
    
    cpu = args.cpu
    if cpu != -1:
        exploration_config['resources']['cpu'] = cpu
    gpu = args.gpu
    if gpu != -1:
        exploration_config['resources']['gpu'] = gpu
    

    
    # ESTABLECER EL CODIGO PARA EL TIPO DE STOPPER
    # DETECTAR TIPO DE STOPPER Y MANDAR


    experiment_info = {
        'max_concurrent': args.max_concurrent,
        'random_state': args.random_state,
        'time_budget': time_budget,
        'restore': args.restore,
        'save_experiment': args.save_experiment
        # 'type_stopper': 
        # 'stopper_custom': args.stopper_custom
    }
    
    # Execute the hyperparameters search
    hyperparameters_search(
        dataset_locations=dataset_locations,
        base_config=base_config,
        exploration_config=exploration_config,
        experiment_full_path=experiment_full_path,
        # time_budget=time_budget,
        experiment_info=experiment_info
    )

# Execute main function
if __name__=="__main__":
    parser = argparse.ArgumentParser(
        prog="Execute experiments in datasets",
        description="Runs experiments in a dataset with a set of configurations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--max_concurrent",
        default=5,
        help="Max number of concurrent executions",
        type=int,
        required=False,
    )
    parser.add_argument(
        "--random_state",
        default=42,
        help="Random state for the experiments",
        type=int,
        required=False,
    )
    parser.add_argument(
        "--time_budget",
        default=-1,
        help="Time budget for the experiments (seconds)",
        type=int,
        required=False,
    )
    parser.add_argument(
        "--stopper_custom",
        default=None,
        help="Set the custom stopper for the experiments: [min, patience]",
        type=int,
        nargs='+',
        required=False,
    )
    parser.add_argument(
        "--dataset_locations_fullpath",
        default="basic/dataset_locations.yaml",
        help="Dataset locations full path",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--data",
        default="../../data",
        help="Dataset locations full path",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--experiment",
        help="Experiment folder",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--cpu",
        help="CPU resources per trial",
        type=float,
        default=-1.0,
        required=False,
    )
    parser.add_argument(
        "--gpu",
        help="GPU resources per trial",
        type=float,
        default=-1.0,
        required=False,
    )
    parser.add_argument(
        "--restore",
        help="Restore the experiment",
        action="store_true",
    )
    parser.add_argument(
        "--save_experiment",
        help="Save experiment files",
        action="store_true",
    )
    # parser.add_argument(
    #     "--experiment_name",
    #     default="Test_experiment",
    #     help="Experiment name",
    #     type=str,
    #     required=False,
    # )

    args = parser.parse_args()
    print(args)
    main(args=args)
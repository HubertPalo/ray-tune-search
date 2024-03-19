from hyperparameters_search import hyperparameters_search
import argparse
from basic.helper import get_dataset_locations
from pathlib import Path
import yaml
import os
from dacite import from_dict
from basic.exploration_config import ExplorationConfig
import traceback


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


    if args.external:
        path = Path(f"{args.experiment}")
    else:
        path = Path(f"experiments/{args.experiment}")
    
    if not os.path.exists(path):
        raise ValueError(f"Experiment path {path} does not exist")
    
    experiment_full_path = Path.absolute(Path(f"{path}"))
    base_config = None
    exploration_config = None
    with open(experiment_full_path /  f"{path}/base_config.yaml", "r") as f:
        base_config = yaml.load(f, Loader=yaml.FullLoader)
    with open(experiment_full_path /  f"{path}/exploration_config.yaml", "r") as f:
        exploration_config = yaml.load(f, Loader=yaml.FullLoader)
    


    # folders_to_check = [
    #     Path(f"{args.experiment}"),
    #     Path(f"experiments/{args.experiment}")
    # ]
    # if args.is_folder:
    #     for folder in os.listdir(args.experiment):
    #         try:
    #             with open(f"{args.experiment}/{folder}/exploration_config.yaml", "r") as f:
    #                 exploration_config = yaml.load(f, Loader=yaml.FullLoader)
    #             with open(f"{args.experiment}/{folder}/base_config.yaml", "r") as f:
    #                 base_config = yaml.load(f, Loader=yaml.FullLoader)
    #             experiment_full_path = Path.absolute(Path(f"{args.experiment}/{folder}"))
    #             break
    #         except Exception:
    #             exploration_config = None
    #             base_config = None
    #             print(f"Files inside path {args.experiment}/{folder} not found...")
    

    # for folder in folders_to_check:
    #     print(f"Looking for files inside path {folder}...")
    #     exploration_config_path = Path.absolute(Path(f"{folder}/exploration_config.yaml"))
    #     base_config_path = Path.absolute(Path(f"{folder}/base_config.yaml"))
    #     try:
    #         print(f"Trying to read file {exploration_config_path}...")
    #         # Read the hyperparameters search config file
    #         with open(exploration_config_path, "r") as f:
    #             exploration_config = yaml.load(f, Loader=yaml.FullLoader)
    #         print(f"Trying to read file {base_config_path}...")
    #         with open(base_config_path, "r") as f:
    #             base_config = yaml.load(f, Loader=yaml.FullLoader)
    #         print(f"Files inside path {folder} found.")
    #         experiment_full_path = Path.absolute(Path(f"{folder}"))
    #         break
    #     except Exception:
    #         print(traceback.format_exc())
    #         exploration_config = None
    #         base_config = None
    #         print(f"Files inside path {args.experiment} not found...")

    if exploration_config is None or base_config is None:
        raise ValueError(f"No experiment files found. Exiting...")
    
    print(exploration_config)
    exploration_config = from_dict(
        data_class=ExplorationConfig,
        data=exploration_config
    )
    
    time_budget = args.time_budget if args.time_budget != -1 else None
    # time_budget = args.time_budget
    # if time_budget == -1:
    #     time_budget = None
    
    exploration_config.resources.cpu = args.cpu if args.cpu != -1 else exploration_config.resources.cpu
    exploration_config.resources.gpu = args.gpu if args.gpu != -1 else exploration_config.resources.gpu
    # cpu = args.cpu
    # if cpu != -1:
    #     exploration_config.resources.cpu = cpu
    # gpu = args.gpu
    # if gpu != -1:
    #     exploration_config.resources.gpu = gpu
    
    # ESTABLECER EL CODIGO PARA EL TIPO DE STOPPER
    # DETECTAR TIPO DE STOPPER Y MANDAR

    experiment_info = {
        'max_concurrent': args.max_concurrent,
        'random_state': args.random_state,
        'time_budget': time_budget,
        'restore': args.restore,
        'save_experiment': args.save_experiment,
        'baseline_gain': args.baseline_gain,
    }
    
    # Execute the hyperparameters search
    hyperparameters_search(
        dataset_locations=dataset_locations,
        base_config=base_config,
        exploration_config=exploration_config,
        experiment_full_path=experiment_full_path,
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
    parser.add_argument(
        "--baseline_gain",
        default="none",
        help="Baseline gain: [none, min, mean]",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--external",
        help="External folder for the experiment files",
        action="store_true",
    )
    # parser.add_argument(
    #     "--is_folder",
    #     default="Test_experiment",
    #     help="Experiment name",
    #     type=str,
    #     required=False,
    # )

    args = parser.parse_args()
    print(args)
    main(args=args)
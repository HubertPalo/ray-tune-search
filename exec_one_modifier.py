import os
import pandas as pd
from ray import tune
import yaml
import argparse
from pathlib import Path

class ConfigEditor:

    def __init__(self, config, filename):
        self.config = config
        self.filename = filename

    def pca_dim_setter(self):
        index_file = self.filename.split(".")[0]
        self.config['reducer']['kwargs']['n_components'] = int(index_file)

    def add_pydrmetrics(self):
        self.config['extra']['report_pydrmetrics'] = True
    
    def execute_function(self, func_name):
        if hasattr(self, func_name) and callable(function_to_call := getattr(self, func_name)):
            function_to_call()


# Main function
def main(args):
    folder_fullpath = Path.absolute(Path(args.folder))
    if not os.path.exists(folder_fullpath):
        raise ValueError(f"Folder {folder_fullpath} does not exist")
    if args.multiply != -1:
        unique_file = os.listdir(folder_fullpath)[0]
        with open(f"{folder_fullpath}/{unique_file}", "rb") as f:
            file_content = yaml.load(f, Loader=yaml.FullLoader)
        for i in range(args.multiply):
            with open(f"{folder_fullpath}/{i}-{unique_file}", "w") as f:
                yaml.dump(file_content, f)
    if args.reset_index:
        files = os.listdir(folder_fullpath)
        for i, file in enumerate(files):
            os.rename(f"{folder_fullpath}/{file}", f"{folder_fullpath}/{i}.yaml")
    if args.func is not None:
        files = os.listdir(folder_fullpath)
        for file in files:
            with open(f"{folder_fullpath}/{file}", "r") as f:
                file_content = yaml.load(f, Loader=yaml.FullLoader)
            config_obj = ConfigEditor(config=file_content, filename = file)
            config_obj.execute_function(args.func)
            with open(f"{folder_fullpath}/{file}", "w") as f:
                yaml.dump(config_obj.config, f)


if __name__=="__main__":
    parser = argparse.ArgumentParser(
        prog="Modify files",
        description="Modify the files for a given experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--folder",
        help="Folder to work in",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--multiply",
        default=-1,
        help="Multiply the file in the folder by the number of times specified",
        type=int,
        required=False,
    )
    parser.add_argument(
        "--reset_index",
        action="store_true",
        help="Change the names of the files in the folder to numbers starting from 0"
    )
    parser.add_argument(
        "--func",
        default=None,
        help="Function to run for every file in the folder",
        type=str,
        required=False,
    )
    args = parser.parse_args()
    print(args)
    main(args=args)


# # Read ray tune experiment results
# with open("config_to_evaluate.yaml", "r") as f:
#     input_file = yaml.load(f, Loader=yaml.FullLoader)



# experiment_name = "umap_hyperparameters_on_kuhar.standartized_balanced_starting_with_30.60.90...300"
# path = os.path.join("../../ray_results", experiment_name)
# restored_tuner = tune.Tuner.restore(path=path, trainable=my_objective_function)
# result_grid = restored_tuner.get_results()
# results_df = pd.DataFrame(result_grid)
# # results_df.to_csv('DATA_EXEMPLO.csv')
# print(results_df)
import os
from datetime import datetime
import yaml
from src.utils.get_folders_utils import get_dataset_name
from pathlib import Path

def create_processed_folder(config):
    project_root_dir = config["project_root_dir"]
    dataset_folder = config["preprocessing"]["dataset_folder"]
    
    dataset_name = get_dataset_name(config)

    processed_data_folder = f"{project_root_dir}/{dataset_folder}/processed/{dataset_name}"
    
    folder_path = Path(processed_data_folder)

    # create folder if it does not exist
    folder_path.mkdir(parents=True, exist_ok=True)
    
    return processed_data_folder


def create_time_folder(config, parent_folder):
    """
    """
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    time_folder = os.path.join(parent_folder, f"exp_{timestamp}")

    os.makedirs(time_folder, exist_ok=True)

    # Save config for reproducibility
    with open(os.path.join(time_folder, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    return time_folder

def create_train_experiments_folder(config):
    project_root_dir = config["project_root_dir"]
    dataset_name = config["preprocessing"]["dataset_name"]
    merge_bre_cu = config["preprocessing"]["merge_bre_cu"]
    model_name = config["training"]["model"]

    name = ""
    if merge_bre_cu: name = "merged"
    else: name = dataset_name

    train_experiments_main_folder = f"{project_root_dir}/train_experiments/{name}/{model_name}"
    
    train_experiments_time_folder = create_time_folder(config, train_experiments_main_folder)

    return train_experiments_time_folder
    
def create_eval_results_folder(config):
    project_root_dir = config["project_root_dir"]
    
    model_name = config["evaluation"]["model"]
    dataset_name = get_dataset_name(config)

    eval_results_folder = f"{project_root_dir}/eval_results/{model_name}/{dataset_name}"

    # Create a folder if it doesn't exist
    os.makedirs(eval_results_folder, exist_ok=True)

    return eval_results_folder



from pathlib import Path
import joblib

def get_dataset_path(config):
    project_root_dir = config["project_root_dir"]
    dataset_name = config["preprocessing"]["dataset_name"]
    dataset_folder = config["preprocessing"]["dataset_folder"]
    data_path = f"{project_root_dir}/{dataset_folder}/{dataset_name}Master.csv"

    path = Path(data_path)
    if not path.is_file():
        raise FileNotFoundError(f"[ERROR] {dataset_name} dataset file does not exist. Please, place datasets into data folder first.")

    return data_path

def get_dataset_name(config):
    merge_bre_cu = config["preprocessing"]["merge_bre_cu"]
    dataset_name = config["preprocessing"]["dataset_name"]

    name = ""
    if merge_bre_cu: name = "merged"
    else: name = dataset_name

    return name

def get_processed_folder(config):
    project_root_dir = config["project_root_dir"]
    dataset_folder = config["preprocessing"]["dataset_folder"]
    
    dataset_name = get_dataset_name(config)

    processed_data_folder = f"{project_root_dir}/{dataset_folder}/processed/{dataset_name}"
    
    path = Path(processed_data_folder)
    if not path.is_dir():
        raise FileNotFoundError(f"[ERROR] Processed data folder does not exist. \
                                Please, make sure to run data preprocessing first.")


    return processed_data_folder

def get_train_experiments_main_folder(config):
    project_root_dir = config["project_root_dir"]
    model_name = config["training"]["model"]

    dataset_name = get_dataset_name(config)

    train_experiments_main_folder = f"{project_root_dir}/train_experiments/{model_name}/{dataset_name}/"

    path = Path(train_experiments_main_folder)
    if not path.is_dir():
        raise FileNotFoundError(f"[ERROR] Processed data folder does not exist. \
                                Please, make sure to run model training first.")
    
    return train_experiments_main_folder

def get_evaluation_results_main_folder(config):
    project_root_dir = config["project_root_dir"]
    model_name = config["evaluation"]["model"]

    dataset_name = get_dataset_name(config)
    evaluation_results_main_folder = f"{project_root_dir}/eval_results/{model_name}/{dataset_name}"

    path = Path(evaluation_results_main_folder)
    if not path.is_dir():
        raise FileNotFoundError(f"[ERROR] Evaluation results data folder does not exist. \
                                Please, make sure to run evaluation first.")
    
    return evaluation_results_main_folder

def get_train_data_saler(config):
    processed_data_folder = get_processed_folder(config)
    scaler = joblib.load(f"{processed_data_folder}/scaler.pkl")
    return scaler
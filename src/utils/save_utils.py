import numpy as np
import joblib
import json
from src.utils.create_folders_utils import create_processed_folder

def save_processed_data(splits_norm, timestamps, scaler, devices, config):
    
    processed_data_folder = create_processed_folder(config)

    # Save arrays
    np.savez(
        f"{processed_data_folder}/arrays.npz",
        train=splits_norm["train"],
        val=splits_norm["val"],
        actor2_test=splits_norm["actor2_test"],
        actor1_test=splits_norm["actor1_test"]
    )

    # Save timestamps
    np.savez(
        f"{processed_data_folder}/timestamps.npz",
        train=timestamps["train"],
        val=timestamps["val"],
        actor2_test=timestamps["actor2_test"],
        actor1_test=timestamps["actor1_test"]
    )

    joblib.dump(scaler, f"{processed_data_folder}/scaler.pkl")

    with open(f"{processed_data_folder}/devices.json", "w") as f:
        json.dump(devices, f)
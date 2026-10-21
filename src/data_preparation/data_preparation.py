import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

from src.utils.downsample import detect_sensor_types
from src.utils.load_actors_timelines import load_actor_timelines
from src.utils.get_folders_utils import get_dataset_path

def filter_columns_one_data(df, config):
    dataset_name = config["preprocessing"]["dataset_name"]
    selected_devices_path = f"configs/{dataset_name}_selected_features.txt"

    print(f"Filtering {dataset_name} data...")

    # read selected devices from txt
    with open(selected_devices_path, "r") as f:
        selected_devices_names = [line.strip() for line in f if line.strip()]

    # keep only selected devices
    selected_devices = [c for c in selected_devices_names if c in df.columns]
    
    # keep timestamp
    if "Timestamp" not in selected_devices:
        selected_devices.append("Timestamp")
    
    # filter dataframe
    df_filtered = df[selected_devices]

    return df_filtered

def filter_columns_merged_data(df):
    print("Filtering BRE+CU merged data...")
    bre_selected_devices_path = "configs/BRE_selected_devices.txt"
    cu_selected_devices_path = "configs/CU_selected_devices.txt"

    # read BRE selected devices
    with open(bre_selected_devices_path, "r") as f:
        bre_selected_devices_names = [line.strip() for line in f if line.strip()]

    # read CU selected devices
    with open(cu_selected_devices_path, "r") as f:
        cu_selected_devices_names = [line.strip() for line in f if line.strip()]

    # merge devices lists
    all_selected_devices = bre_selected_devices_names + cu_selected_devices_names

    # remove duplicates
    all_selected_devices = list(set(all_selected_devices))

    # keep timestamp
    if "Timestamp" not in all_selected_devices:
        all_selected_devices.append("Timestamp")

    # keep only actuall existing devices in the raw data
    selected_columns = [c for c in all_selected_devices if c in df.columns]

    # filter dataframe
    df_filtered = df[selected_columns]

    return df_filtered

def clean_cu_data(df):
    """
    CU dataset has the column "light.philips_hue_lightstrip_pid_146 " 
    with white sapce at the end
    CU dataset has the column "light.philips_hue_lightstrip_pid_146 " with all NaN
    By default, df.dropna() removes any row that has at least one NaN. (we dont want that)
    """
    print("Cleaning CU data...")
    df.columns = df.columns.str.strip()
    
    # Replace fake values with NaN
    # e.g., sensor.aqara_wireless_switch_pid_081_action → mean = -9.68
    # df.replace([-9.21, -9.68, -9.7, -9.81], pd.NA, inplace=True)
    
    # separate timestamp
    # timestamp_col = None

    # if "Timestamp" in df.columns:
    #     timestamp_col = df["Timestamp"]
    #     df = df.drop(columns=["Timestamp"])

    # keep only numeric columns
    # df = df.select_dtypes(include=['number'])

    # remove near-constant sensors
    # e.g., switch.kasa_023 → std = 0, this brings ZERO information
    # df = df.loc[:, df.std() > 1e-6]

    # restore timestamp
    # if timestamp_col is not None:
    #     df.insert(0, "Timestamp", timestamp_col)

    df = df.dropna(axis=1, how="all")   # remove dead sensors
    # df = df.fillna(method="ffill")      # or interpolate
    return df

def downsample_data(df, downsample_freq):
    """
    Downsample data to target frequency while preserving events.
    continuous sensors → mean
    binary/event sensors → max
    """
    print(f"Downsampling data to {downsample_freq}...")

    # Ensure Timestamp exists
    if "Timestamp" not in df.columns:
        raise ValueError("Timestamp column not found in dataframe")

    # Convert timestamp
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    # Set index
    df = df.set_index("Timestamp")

    # Detect sensor types AFTER timestamp handling
    binary_cols, continuous_cols = detect_sensor_types(df)

    binary_cols = binary_cols or []
    continuous_cols = continuous_cols or []

    # aggregation dictionary
    agg_dict = {}

    for col in continuous_cols:
        agg_dict[col] = "mean"

    for col in binary_cols:
        agg_dict[col] = "max"

    # downsample
    df_down = df.resample(downsample_freq).agg(agg_dict)

    df_down = df_down.reset_index()

    return df_down

def split_actor_periods(df, config):
    """
    Split dataset into train/val/test according to actor timelines.
    - actor1_train (normal training from Actor 1 timeline 1 only)
    - actor1_val (normal validation from Actor 1 timeline 1 only)
    - actor2_test (test from Actor 2 timeline)
    - act
    """
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    # Load timelines and val_ratio from config
    timelines = load_actor_timelines(config)
    val_ratio = config["preprocessing"]["val_ratio"]

    # Slice data according to timelines
    actor1_t1 = df[
        (df["Timestamp"] >= timelines["actor1_t1"][0]) &
        (df["Timestamp"] <= timelines["actor1_t1"][1])
    ]

    actor2 = df[
        (df["Timestamp"] >= timelines["actor2"][0]) &
        (df["Timestamp"] <= timelines["actor2"][1])
    ]

    actor1_t2 = df[
        (df["Timestamp"] >= timelines["actor1_t2"][0]) &
        (df["Timestamp"] <= timelines["actor1_t2"][1])
    ]

    # Sort
    actor1_t1 = actor1_t1.sort_values("Timestamp")
    actor2_test_df = actor2.sort_values("Timestamp")
    actor1_test_df = actor1_t2.sort_values("Timestamp")

    # Split train/val
    split_idx = int(len(actor1_t1) * (1 - val_ratio))
    train_df = actor1_t1.iloc[:split_idx]
    val_df   = actor1_t1.iloc[split_idx:]

    splits = { "train": train_df, 
               "val": val_df, 
               "actor2_test": actor2_test_df, 
               "actor1_test": actor1_test_df
            }
    
    return splits

def normalize(splits, devices):
    """
    Normalize data using ONLY train_df (Actor1 timeline1)
    Returns:
    - train/val/test arrays (features only)
    - scaler
    """
    scaler = MinMaxScaler()

    # Fit scaler only on training data (features only without timestamp)
    # .to_numpy() is safer than .values() which removes column structure
    train_array_norm = scaler.fit_transform(splits["train"][devices].to_numpy())
    val_array_norm   = scaler.transform(splits["val"][devices].to_numpy())
    actor2_test_array_norm = scaler.transform(splits["actor2_test"][devices].to_numpy())
    actor1_test_array_norm = scaler.transform(splits["actor1_test"][devices].to_numpy())

    splits_norm = {
        "train":train_array_norm,
        "val": val_array_norm,
        "actor2_test":actor2_test_array_norm,
        "actor1_test": actor1_test_array_norm
    }
    return splits_norm, scaler

def load_one_data(config):
    data_path = get_dataset_path(config)
    dataset_name = config["preprocessing"]["dataset_name"]
    
    print(f"Loading {dataset_name} data...")
    df = pd.read_csv(data_path)
    return df

def load_and_merge_bre_cu(config):
    
    project_root_dir = config["project_root_dir"]
    dataset_folder = config["preprocessing"]["dataset_folder"]

    bre_data_path = f"{project_root_dir}/{dataset_folder}/BREMaster.csv"
    cu_data_path = f"{project_root_dir}/{dataset_folder}/CUMaster.csv"

    path = Path(bre_data_path)
    if not path.is_file():
        raise FileNotFoundError(f"[ERROR] Data file does not exist. Please, place datasets into data folder first.")
    
    path = Path(cu_data_path)
    if not path.is_file():
        raise FileNotFoundError(f"[ERROR] Data file does not exist. Please, place datasets into data folder first.")
    
    print("Loading BRE dataset...")
    bre_df = pd.read_csv(bre_data_path)

    print("Loading CU dataset...")
    cu_df = pd.read_csv(cu_data_path)

    cu_df = clean_cu_data(cu_df)
    # Convert timestamp
    bre_df["Timestamp"] = pd.to_datetime(bre_df["Timestamp"])
    cu_df["Timestamp"] = pd.to_datetime(cu_df["Timestamp"])

    # Sort for time series merge
    bre_df = bre_df.sort_values("Timestamp")
    cu_df = cu_df.sort_values("Timestamp")

    print("Merging BRE+CU data...")

    merged = pd.merge_asof(
        bre_df,
        cu_df,
        on="Timestamp",
        direction="nearest"
    )

    return merged

def preprocessing_pipeline(config):
    """
    Full preprocessing pipeline for GDN:
    1. Load CSV data (either BRE or CU or both merged)
    2. OPTIONAL: Filter data: keep selected features only 
    3. OPTIONAL: Clean data (CU only)
    4. OPTIONAL: Downsample data to target frequency
    5. Split actors timelines (train/val/test)
    6. Normalize/Scale features (devices)
    
    Returns:
    - arrays.npz: contains train/val/actor2_test/actor1_test splits (features only)
    - timestamps.npz: timestamps arrays for each split (for plotting/reference)
    - scaler
    - devices.json: contains devices list
    """
   
    merge_bre_cu = config["preprocessing"]["merge_bre_cu"]
    dataset_name = config["preprocessing"]["dataset_name"]

    # 1. Load CSV data (either BRE or CU or both merged)
    if merge_bre_cu: df = load_and_merge_bre_cu(config) # both BRE and CU
    else: df = load_one_data(config) # one data 
    
    # 2. Filter data (if enabled): keep selected features only
    apply_filtering = config["preprocessing"]["apply_filtering"]
    if apply_filtering:
        if merge_bre_cu: df = filter_columns_merged_data(df)
        else: df = filter_columns_one_data(df, config)

    # 2. Clean CU data if loaded alone (otherwise it will cleaned in merging function)
    apply_cleaning = config["preprocessing"]["apply_cleaning"]
    if apply_cleaning:
        if not(merge_bre_cu) and dataset_name == "CU": df = clean_cu_data(df)
    
    # 4. Downsample data to target frequency if not 1s
    downsample_freq = config["preprocessing"]["downsample_freq"]
    if downsample_freq != "1s": df = downsample_data(df, downsample_freq)
    
    # 5. Get device columns (exclude Timestamp)
    devices = [c for c in df.columns if c != "Timestamp"]
    print("nbr of devices:", len(devices))

    # 6. Split actors / train-val-test
    splits = split_actor_periods(df, config)
    print("Actor split done.")

    # 7. Save timestamps for reference/plotting
    train_timestamps = splits["train"]['Timestamp'].to_numpy()
    val_timestamps   = splits["val"]['Timestamp'].to_numpy()
    actor2_test_timestamps = splits["actor2_test"]['Timestamp'].to_numpy()
    actor1_test_timestamps = splits["actor1_test"]['Timestamp'].to_numpy()

    timestamps = {
        "train": train_timestamps,
        "val": val_timestamps,
        "actor2_test": actor2_test_timestamps,
        "actor1_test": actor1_test_timestamps
    }
    
    # 8. Normalize features
    splits_norm, scaler = normalize(splits, devices)
    print("Normalization done.")

    print("Train split:", len(splits_norm["train"]), f'% {len(splits_norm["train"])/len(df)*100:.2f}') 
    print("Validation split:",len(splits_norm["val"]), f'% {len(splits_norm["val"])/len(df)*100:.2f}')
    print("Actor 2 test split:",len(splits_norm["actor2_test"]), f'% {len(splits_norm["actor2_test"])/len(df)*100:.2f}')
    print("Actor 1 test split:",len(splits_norm["actor1_test"]), f'% {len(splits_norm["actor1_test"])/len(df)*100:.2f}')

    return (splits_norm, timestamps, scaler, devices)
import os
import pickle
from urllib.parse import urlparse

import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def resolve_db_config(db_cfg, env_prefix="DB"):
    """Config values can be overridden by env vars, e.g. when running inside Docker
    or on Render, where hosts are container/private-network names instead of localhost.

    A single f"{env_prefix}_URL" (postgresql://user:pass@host:port/db, as provided by
    Render's `fromDatabase: property: connectionString`) takes precedence over the
    discrete _HOST/_PORT/etc. env vars when set."""
    url = os.environ.get(f"{env_prefix}_URL")
    if url:
        parsed = urlparse(url)
        return {
            "host": parsed.hostname,
            "port": parsed.port or db_cfg["port"],
            "dbname": parsed.path.lstrip("/"),
            "user": parsed.username,
            "password": parsed.password,
            "table": db_cfg["table"],
        }
    return {
        "host": os.environ.get(f"{env_prefix}_HOST", db_cfg["host"]),
        "port": os.environ.get(f"{env_prefix}_PORT", db_cfg["port"]),
        "dbname": os.environ.get(f"{env_prefix}_NAME", db_cfg["dbname"]),
        "user": os.environ.get(f"{env_prefix}_USER", db_cfg["user"]),
        "password": os.environ.get(f"{env_prefix}_PASSWORD", db_cfg["password"]),
        "table": db_cfg["table"],
    }


def get_db_engine(db_cfg):
    from sqlalchemy import create_engine
    url = f"postgresql+psycopg2://{db_cfg['user']}:{db_cfg['password']}@{db_cfg['host']}:{db_cfg['port']}/{db_cfg['dbname']}"
    return create_engine(url)


def load_dataframe(data_cfg):
    """Loads the labeled transaction data either from a file snapshot or the live warehouse."""
    mode = data_cfg.get("mode", "file")

    if mode == "file":
        path = data_cfg["file_path"]
        if path.endswith(".parquet"):
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
    elif mode == "db":
        db = resolve_db_config(data_cfg["db"])
        engine = get_db_engine(db)
        df = pd.read_sql_table(db["table"], engine)
    else:
        raise ValueError(f"Unknown data mode: {mode}")

    drop_columns = [c for c in data_cfg.get("drop_columns", []) if c in df.columns]
    return df.drop(columns=drop_columns)


class FraudTabularDataset(Dataset):
    """Wraps pre-encoded categorical/continuous tensors and binary fraud labels."""

    def __init__(self, x_cat, x_cont, y):
        self.x_cat = torch.as_tensor(x_cat, dtype=torch.long)
        self.x_cont = torch.as_tensor(x_cont, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x_cat[idx], self.x_cont[idx], self.y[idx]


def build_dataloaders(config):
    data_cfg = config["data"]
    df = load_dataframe(data_cfg)

    target_col = data_cfg["target_column"]
    cat_cols = data_cfg["categorical_columns"]
    cont_cols = data_cfg["continuous_columns"]

    y = df[target_col].astype(int).values

    # Label-encode categorical columns; each column's cardinality feeds the model's embeddings.
    encoders = {}
    cat_cardinalities = []
    x_cat = np.zeros((len(df), len(cat_cols)), dtype=np.int64)
    for i, col in enumerate(cat_cols):
        encoder = LabelEncoder()
        x_cat[:, i] = encoder.fit_transform(df[col].astype(str))
        encoders[col] = encoder
        cat_cardinalities.append(len(encoder.classes_))

    x_cont = df[cont_cols].astype(float).fillna(0.0).values

    seed = data_cfg.get("random_seed", 42)
    test_size = data_cfg.get("test_size", 0.15)
    val_size = data_cfg.get("val_size", 0.15)

    idx = np.arange(len(df))

    def _safe_stratify(labels, n_splits_needed):
        # Stratification requires every class to have enough members for each resulting split.
        counts = np.bincount(labels)
        if counts[counts > 0].min() < n_splits_needed:
            return None
        return labels

    stratify_first = _safe_stratify(y, 2)
    train_idx, temp_idx = train_test_split(
        idx, test_size=test_size + val_size, random_state=seed, stratify=stratify_first
    )
    relative_val_size = val_size / (test_size + val_size)
    stratify_second = _safe_stratify(y[temp_idx], 2)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=1 - relative_val_size, random_state=seed, stratify=stratify_second
    )

    # Fit the continuous-feature scaler on train only, then apply to all splits.
    scaler = StandardScaler()
    x_cont_train = scaler.fit_transform(x_cont[train_idx])
    x_cont_val = scaler.transform(x_cont[val_idx])
    x_cont_test = scaler.transform(x_cont[test_idx])

    batch_size = config["training"]["batch_size"]

    train_ds = FraudTabularDataset(x_cat[train_idx], x_cont_train, y[train_idx])
    val_ds = FraudTabularDataset(x_cat[val_idx], x_cont_val, y[val_idx])
    test_ds = FraudTabularDataset(x_cat[test_idx], x_cont_test, y[test_idx])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    meta = {
        "cat_cardinalities": cat_cardinalities,
        "num_continuous": len(cont_cols),
        "encoders": encoders,
        "scaler": scaler,
    }
    return train_loader, val_loader, test_loader, meta


def save_inference_meta(meta, cat_cols, cont_cols, path):
    """Persists everything the live serving loop needs to reproduce training-time
    encoding (label encoders + scaler) without re-running build_dataloaders."""
    payload = {
        "cat_cols": cat_cols,
        "cont_cols": cont_cols,
        "cat_cardinalities": meta["cat_cardinalities"],
        "encoders": meta["encoders"],
        "scaler": meta["scaler"],
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load_inference_meta(path):
    with open(path, "rb") as f:
        return pickle.load(f)

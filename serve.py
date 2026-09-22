"""Live fraud scoring loop.

Continuously polls the training Postgres db (see config.yaml `data.db`) for
transactions that haven't been scored yet, runs them through the current best
checkpoint, and persists the predicted fraud probability/label back to Postgres
(`serving.predictions_table`) so the measure survives restarts and is queryable
independent of this process.

Run standalone: `python serve.py [--once]`
"""
import argparse
import logging
import os
import time

import numpy as np
import pandas as pd
import torch
from sqlalchemy import text

from dataset import load_config, load_inference_meta, resolve_db_config, get_db_engine
from engine import resolve_device
from model import TabularTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def ensure_predictions_table(engine, table):
    with engine.begin() as conn:
        conn.execute(text(
            f"CREATE TABLE IF NOT EXISTS {table} ("
            "event_id TEXT PRIMARY KEY, "
            "fraud_probability DOUBLE PRECISION, "
            "predicted_is_fraud INTEGER, "
            "scored_at TIMESTAMP DEFAULT now())"
        ))


def fetch_unscored_rows(engine, events_table, predictions_table, batch_size):
    query = text(
        f"SELECT e.* FROM {events_table} e "
        f"LEFT JOIN {predictions_table} p ON e.event_id = p.event_id "
        f"WHERE p.event_id IS NULL "
        f"ORDER BY e.event_timestamp LIMIT :batch_size"
    )
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params={"batch_size": batch_size})


def encode_batch(df, inference_meta):
    cat_cols, cont_cols = inference_meta["cat_cols"], inference_meta["cont_cols"]
    encoders = inference_meta["encoders"]

    x_cat = np.zeros((len(df), len(cat_cols)), dtype=np.int64)
    for i, col in enumerate(cat_cols):
        encoder = encoders[col]
        known = set(encoder.classes_)
        # Categories unseen during training fall back to a known class so live
        # scoring never crashes on new/unexpected values.
        values = df[col].astype(str).where(df[col].astype(str).isin(known), encoder.classes_[0])
        x_cat[:, i] = encoder.transform(values)

    x_cont = inference_meta["scaler"].transform(df[cont_cols].astype(float).fillna(0.0).values)
    return x_cat, x_cont


@torch.no_grad()
def score_batch(model, device, x_cat, x_cont):
    x_cat_t = torch.as_tensor(x_cat, dtype=torch.long, device=device)
    x_cont_t = torch.as_tensor(x_cont, dtype=torch.float32, device=device)
    logits = model(x_cat_t, x_cont_t).squeeze(1)
    return torch.sigmoid(logits).cpu().numpy()


def persist_predictions(engine, table, event_ids, probabilities, threshold=0.5):
    records = [
        {"event_id": eid, "fraud_probability": float(p), "predicted_is_fraud": int(p >= threshold)}
        for eid, p in zip(event_ids, probabilities)
    ]
    stmt = text(
        f"INSERT INTO {table} (event_id, fraud_probability, predicted_is_fraud) "
        f"VALUES (:event_id, :fraud_probability, :predicted_is_fraud) "
        f"ON CONFLICT (event_id) DO NOTHING"
    )
    with engine.begin() as conn:
        conn.execute(stmt, records)


def load_model(config, inference_meta, device):
    model_cfg = config["model"]
    model = TabularTransformer(
        cat_cardinalities=inference_meta["cat_cardinalities"],
        num_continuous=len(inference_meta["cont_cols"]),
        dim=model_cfg.get("dim", 32),
        depth=model_cfg.get("depth", 4),
        heads=model_cfg.get("heads", 4),
        dim_feedforward=model_cfg.get("dim_feedforward", 64),
        dropout=model_cfg.get("dropout", 0.1),
    )
    checkpoint_dir = config["training"].get("checkpoint_dir", "checkpoints")
    state = torch.load(os.path.join(checkpoint_dir, "best_model.pt"), map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Continuously score new transactions and persist fraud predictions.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--once", action="store_true", help="Run a single scoring pass and exit.")
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config["data"]
    if data_cfg.get("mode") != "db":
        raise ValueError("Live scoring requires data.mode: db in config.yaml.")

    serving_cfg = config.get("serving", {})
    poll_interval = serving_cfg.get("poll_interval_seconds", 15)
    batch_size = serving_cfg.get("batch_size", 256)
    predictions_table = serving_cfg.get("predictions_table", "fraud_predictions")

    checkpoint_dir = config["training"].get("checkpoint_dir", "checkpoints")
    inference_meta = load_inference_meta(os.path.join(checkpoint_dir, "inference_meta.pkl"))

    device = resolve_device(config["training"].get("device", "auto"))
    model = load_model(config, inference_meta, device)

    db_cfg = resolve_db_config(data_cfg["db"])
    engine = get_db_engine(db_cfg)
    ensure_predictions_table(engine, predictions_table)

    logger.info("Live scoring started | polling every %ss | writing predictions to %s",
                poll_interval, predictions_table)

    while True:
        df = fetch_unscored_rows(engine, db_cfg["table"], predictions_table, batch_size)
        if df.empty:
            logger.info("No unscored transactions.")
        else:
            x_cat, x_cont = encode_batch(df, inference_meta)
            probs = score_batch(model, device, x_cat, x_cont)
            persist_predictions(engine, predictions_table, df["event_id"].tolist(), probs)
            logger.info("Scored and persisted %d transactions.", len(df))

        if args.once:
            break
        time.sleep(poll_interval)


if __name__ == "__main__":
    main()

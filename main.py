import argparse
import logging
import os
import random
import time

import numpy as np
import torch

from dataset import load_config, load_dataframe, build_dataloaders, save_inference_meta
from drift import compute_reference_stats, save_reference_stats, load_reference_stats, has_drifted
from model import TabularTransformer
from engine import train, evaluate, FocalLoss, resolve_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_training_cycle(config):
    data_cfg = config["data"]
    logger.info("Building dataloaders (mode=%s)...", data_cfg.get("mode", "file"))
    train_loader, val_loader, test_loader, meta = build_dataloaders(config)
    logger.info(
        "Loaded %d train / %d val / %d test rows | %d categorical features / %d continuous features",
        len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset),
        len(meta["cat_cardinalities"]), meta["num_continuous"],
    )

    model_cfg = config["model"]
    model = TabularTransformer(
        cat_cardinalities=meta["cat_cardinalities"],
        num_continuous=meta["num_continuous"],
        dim=model_cfg.get("dim", 32),
        depth=model_cfg.get("depth", 4),
        heads=model_cfg.get("heads", 4),
        dim_feedforward=model_cfg.get("dim_feedforward", 64),
        dropout=model_cfg.get("dropout", 0.1),
    )
    logger.info("Model built. Starting training...")

    model, device = train(model, train_loader, val_loader, config)
    logger.info("Training complete on device=%s. Evaluating on test set...", device)

    training_cfg = config["training"]
    criterion = FocalLoss(
        alpha=training_cfg.get("focal_loss_alpha", 0.25),
        gamma=training_cfg.get("focal_loss_gamma", 2.0),
    )
    test_metrics = evaluate(model, test_loader, criterion, device)
    logger.info(
        "Test | loss=%.4f | roc_auc=%.4f | pr_auc=%.4f | f1=%.4f",
        test_metrics["loss"], test_metrics["roc_auc"], test_metrics["pr_auc"], test_metrics["f1"],
    )

    checkpoint_dir = training_cfg.get("checkpoint_dir", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_inference_meta(meta, data_cfg["categorical_columns"], data_cfg["continuous_columns"],
                         os.path.join(checkpoint_dir, "inference_meta.pkl"))


def main():
    parser = argparse.ArgumentParser(description="Train the tabular transformer, retraining on data drift.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--once", action="store_true", help="Run a single drift-check/train pass and exit.")
    args = parser.parse_args()

    logger.info("Loading config.yaml...")
    config = load_config(args.config)
    set_seed(config["data"].get("random_seed", 42))

    data_cfg = config["data"]
    cat_cols = data_cfg["categorical_columns"]
    cont_cols = data_cfg["continuous_columns"]

    drift_cfg = config.get("drift", {})
    drift_enabled = drift_cfg.get("enabled", False)
    threshold = drift_cfg.get("threshold", 0.25)
    stats_path = drift_cfg.get("reference_stats_path", "checkpoints/reference_stats.json")
    check_interval = drift_cfg.get("check_interval_seconds", 300)

    while True:
        if drift_enabled:
            df = load_dataframe(data_cfg)
            reference_stats = load_reference_stats(stats_path)
            drifted, score, _ = has_drifted(reference_stats, df, cat_cols, cont_cols, threshold)
            if not drifted:
                logger.info("No significant drift detected (score=%.4f < threshold=%.4f); skipping retrain.",
                             score, threshold)
            else:
                logger.info("Drift threshold crossed (score=%s >= %.4f); retraining...", score, threshold)
                run_training_cycle(config)
                save_reference_stats(compute_reference_stats(df, cat_cols, cont_cols), stats_path)
        else:
            run_training_cycle(config)

        if args.once:
            break
        time.sleep(check_interval)


if __name__ == "__main__":
    main()


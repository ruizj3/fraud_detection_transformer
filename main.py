import logging
import random

import numpy as np
import torch

from dataset import load_config, build_dataloaders
from model import TabularTransformer
from engine import train, evaluate, FocalLoss, resolve_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    logger.info("Loading config.yaml...")
    config = load_config("config.yaml")
    set_seed(config["data"].get("random_seed", 42))

    logger.info("Loading data and building dataloaders (mode=%s)...", config["data"].get("mode", "file"))
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


if __name__ == "__main__":
    main()

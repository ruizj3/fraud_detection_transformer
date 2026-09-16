"""Non-transformer baseline for comparison against the tabular transformer model.

Gradient-boosted trees are a strong, fast baseline for tabular fraud detection and
handle the categorical/continuous mix without needing embeddings or scaling.
"""
import logging

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.model_selection import train_test_split

from dataset import load_config, load_dataframe

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def main():
    logger.info("Loading config.yaml...")
    config = load_config("config.yaml")
    data_cfg = config["data"]

    logger.info("Loading data (mode=%s)...", data_cfg.get("mode", "file"))
    df = load_dataframe(data_cfg)
    target_col = data_cfg["target_column"]
    cat_cols = data_cfg["categorical_columns"]
    cont_cols = data_cfg["continuous_columns"]
    feature_cols = cat_cols + cont_cols
    logger.info("Loaded %d rows | %d categorical / %d continuous features", len(df), len(cat_cols), len(cont_cols))

    for col in cat_cols:
        df[col] = df[col].astype("category")

    x = df[feature_cols]
    y = df[target_col].astype(int).values

    seed = data_cfg.get("random_seed", 42)
    test_size = data_cfg.get("test_size", 0.15)
    val_size = data_cfg.get("val_size", 0.15)

    logger.info("Splitting into train/val/test...")
    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=test_size + val_size, random_state=seed, stratify=y
    )
    relative_val_size = val_size / (test_size + val_size)
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=1 - relative_val_size, random_state=seed, stratify=y_temp
    )
    logger.info("Train=%d | Val=%d | Test=%d", len(x_train), len(x_val), len(x_test))

    categorical_mask = [col in cat_cols for col in feature_cols]
    model = HistGradientBoostingClassifier(
        categorical_features=categorical_mask,
        max_iter=300,
        learning_rate=0.05,
        early_stopping=True,
        random_state=seed,
    )
    logger.info("Training HistGradientBoostingClassifier...")
    model.fit(x_train, y_train)
    logger.info("Training complete. Evaluating...")

    for split_name, x_split, y_split in [("val", x_val, y_val), ("test", x_test, y_test)]:
        probs = model.predict_proba(x_split)[:, 1]
        preds = (probs >= 0.5).astype(int)
        logger.info(
            "%s | roc_auc=%.4f | pr_auc=%.4f | f1=%.4f",
            split_name,
            roc_auc_score(y_split, probs),
            average_precision_score(y_split, probs),
            f1_score(y_split, preds, zero_division=0),
        )


if __name__ == "__main__":
    main()

"""Periodically ingest labeled transactions from local-gcp-mirror-pipeline's live
Postgres warehouse (fraud_events) into this repo's own Postgres db, so `dataset.py`
can read a stable, independently-versioned source for train/test splitting without
needing the upstream pipeline running at training time.

Runs standalone (`python ingest.py --once`) or continuously inside the `ingester`
Docker Compose service, polling on `ingest.poll_interval_seconds`.
"""
import argparse
import logging
import os
import time
from urllib.parse import urlparse

import pandas as pd
import yaml
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# Column order/types match local-gcp-mirror-pipeline's fraud_events table.
SCHEMA = {
    "event_id": "TEXT PRIMARY KEY",
    "event_timestamp": "TIMESTAMP",
    "user_id": "TEXT",
    "amount": "DOUBLE PRECISION",
    "merchant_category": "TEXT",
    "device": "TEXT",
    "is_new_device": "BOOLEAN",
    "payment_method": "TEXT",
    "country": "TEXT",
    "ip_country": "TEXT",
    "account_age_days": "INTEGER",
    "time_since_last_txn_seconds": "DOUBLE PRECISION",
    "txn_count_last_1h": "INTEGER",
    "distance_from_home_km": "DOUBLE PRECISION",
    "is_fraud": "INTEGER",
    "fraud_scenario": "TEXT",
}


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def resolve_db_config(db_cfg, env_prefix):
    """Config values can be overridden by env vars, e.g. when running inside Docker
    or on Render, where hosts are container/private-network names instead of localhost.

    A single f"{env_prefix}_URL" (as provided by Render's `fromDatabase: property:
    connectionString`) takes precedence over the discrete _HOST/_PORT/etc. env vars."""
    url = os.environ.get(f"{env_prefix}_URL")
    if url:
        parsed = urlparse(url)
        return {
            "host": parsed.hostname,
            "port": parsed.port or db_cfg["port"],
            "dbname": parsed.path.lstrip("/"),
            "user": parsed.username,
            "password": parsed.password,
        }
    return {
        "host": os.environ.get(f"{env_prefix}_HOST", db_cfg["host"]),
        "port": os.environ.get(f"{env_prefix}_PORT", db_cfg["port"]),
        "dbname": os.environ.get(f"{env_prefix}_NAME", db_cfg["dbname"]),
        "user": os.environ.get(f"{env_prefix}_USER", db_cfg["user"]),
        "password": os.environ.get(f"{env_prefix}_PASSWORD", db_cfg["password"]),
    }


def get_engine(db_cfg):
    url = (
        f"postgresql+psycopg2://{db_cfg['user']}:{db_cfg['password']}"
        f"@{db_cfg['host']}:{db_cfg['port']}/{db_cfg['dbname']}"
    )
    return create_engine(url)


def ensure_dest_table(engine, table):
    columns_sql = ",\n    ".join(f"{col} {ddl}" for col, ddl in SCHEMA.items())
    with engine.begin() as conn:
        conn.execute(text(f"CREATE TABLE IF NOT EXISTS {table} (\n    {columns_sql}\n)"))


def get_watermark(engine, table):
    """Highest event_timestamp already ingested, so re-polling only pulls new rows."""
    with engine.begin() as conn:
        return conn.execute(text(f"SELECT max(event_timestamp) FROM {table}")).scalar()


def fetch_new_rows(source_engine, source_table, watermark):
    columns = ", ".join(SCHEMA.keys())
    if watermark is None:
        query = text(f"SELECT {columns} FROM {source_table} ORDER BY event_timestamp")
        params = {}
    else:
        query = text(
            f"SELECT {columns} FROM {source_table} WHERE event_timestamp > :watermark ORDER BY event_timestamp"
        )
        params = {"watermark": watermark}
    with source_engine.connect() as conn:
        return pd.read_sql(query, conn, params=params)


def ingest_once(source_engine, dest_engine, source_table, dest_table):
    watermark = get_watermark(dest_engine, dest_table)
    df = fetch_new_rows(source_engine, source_table, watermark)
    if df.empty:
        logger.info("No new rows in %s past watermark %s", source_table, watermark)
        return 0

    columns = list(df.columns)
    insert_cols = ", ".join(columns)
    insert_placeholders = ", ".join(f":{c}" for c in columns)
    # event_id is the primary key, so an overlapping re-pull is a safe no-op.
    stmt = text(
        f"INSERT INTO {dest_table} ({insert_cols}) VALUES ({insert_placeholders}) "
        f"ON CONFLICT (event_id) DO NOTHING"
    )
    records = df.astype(object).where(pd.notnull(df), None).to_dict(orient="records")
    with dest_engine.begin() as conn:
        conn.execute(stmt, records)
    logger.info(
        "Ingested %d new rows from %s (watermark was %s) into %s", len(records), source_table, watermark, dest_table
    )
    return len(records)


def main():
    parser = argparse.ArgumentParser(
        description="Periodically ingest labeled transactions from the pipeline's Postgres warehouse."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--once", action="store_true", help="Run a single ingestion pass and exit.")
    args = parser.parse_args()

    config = load_config(args.config)
    ingest_cfg = config["ingest"]
    source_cfg = resolve_db_config(ingest_cfg["source"], "SOURCE_DB")
    dest_cfg = resolve_db_config(ingest_cfg["db"], "DEST_DB")
    source_table = ingest_cfg["source"]["table"]
    dest_table = ingest_cfg["db"]["table"]
    interval = ingest_cfg.get("poll_interval_seconds", 60)

    source_engine = get_engine(source_cfg)
    dest_engine = get_engine(dest_cfg)
    ensure_dest_table(dest_engine, dest_table)

    logger.info(
        "Ingesting %s@%s:%s -> %s@%s:%s",
        source_table, source_cfg["host"], source_cfg["port"],
        dest_table, dest_cfg["host"], dest_cfg["port"],
    )

    if args.once:
        ingest_once(source_engine, dest_engine, source_table, dest_table)
        return

    while True:
        ingest_once(source_engine, dest_engine, source_table, dest_table)
        time.sleep(interval)


if __name__ == "__main__":
    main()

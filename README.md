# Running the full pipeline
cd local-gcp-mirror-pipeline && docker-compose up -d
python kafka/mock_producer.py            # generates labeled transactions
python spark/jobs/streaming_ingest.py     # streams into Postgres fraud_events
python scripts/export_training_data.py --output-dir ../fraud_detection_transformer/data

cd ../fraud_detection_transformer
pip install -r requirements.txt
python main.py        # train the transformer
python baseline.py    # train/evaluate the gradient-boosting baseline
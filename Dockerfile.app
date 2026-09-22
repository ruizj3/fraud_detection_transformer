FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY dataset.py drift.py engine.py model.py main.py serve.py .

CMD ["python", "main.py"]

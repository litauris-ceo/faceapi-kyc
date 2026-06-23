FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 curl unzip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /root/.insightface/models && \
    curl -L -o /root/.insightface/models/buffalo_l.zip \
    https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip && \
    cd /root/.insightface/models && unzip buffalo_l.zip -d buffalo_l && rm buffalo_l.zip

COPY app.py .

EXPOSE 10000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]

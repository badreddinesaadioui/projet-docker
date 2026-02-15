FROM python:3.9-slim

WORKDIR /train

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train.py .

ENV DATA_DIR=/data
ENV MODEL_DIR=/model

CMD ["python", "train.py"]

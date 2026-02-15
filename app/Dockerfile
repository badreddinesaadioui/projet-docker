FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY templates/ templates/

ENV MODEL_DIR=/model
ENV FLASK_APP=app.py

EXPOSE 5000

CMD ["python", "-u", "app.py"]

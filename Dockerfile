FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache /tmp/*

COPY app.py compliance_data.zip ./

ENV SPACE_ID=local-docker
EXPOSE 7860

CMD ["python", "app.py"]

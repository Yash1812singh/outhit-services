# One image recipe for every service: docker build --build-arg SERVICE=market_data --build-arg PORT=8201 .
FROM python:3.12-slim
ARG SERVICE
ARG PORT=8000
ENV SERVICE=${SERVICE} PORT=${PORT} PYTHONPATH=/app PYTHONUNBUFFERED=1
WORKDIR /app
COPY ${SERVICE}/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY outhit_common /app/outhit_common
COPY ${SERVICE} /app/${SERVICE}
WORKDIR /app/${SERVICE}
RUN mkdir -p data
EXPOSE ${PORT}
CMD ["sh", "-c", "exec uvicorn ${SERVICE}.main:app --host 0.0.0.0 --port ${PORT} --timeout-graceful-shutdown 3"]

FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 MLFLOW_DISABLE_AGENT_HINT=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Generate data and precompute rule-only results so the image works without an API key.
RUN python -m scripts.generate_data && python -m evaluation.evaluate --modes rules --no-mlflow
EXPOSE 8000 8501 5000
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

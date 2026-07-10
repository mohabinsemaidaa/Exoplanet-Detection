FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt

COPY src/ src/
COPY api/ api/
COPY app/ app/
COPY reports/ reports/
COPY data/processed/ data/processed/

EXPOSE 8000 8501 7860

CMD ["streamlit", "run", "app/dashboard.py", "--server.address", "0.0.0.0", "--server.port", "7860"]

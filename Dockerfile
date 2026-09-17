FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p data/uploads data/indexes data/cache data/exports

EXPOSE 8000
CMD ["uvicorn", "knowledge_assistant.main:app", "--host", "0.0.0.0", "--port", "8000"]
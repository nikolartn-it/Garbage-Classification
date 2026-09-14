FROM python:3.12-slim

WORKDIR /app

# Sistemske zavisnosti (potrebne za torch i pillow)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Instaliraj Python zavisnosti
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Kopiraj source code
COPY src/ src/
COPY models/ models/
COPY app.py .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# ============================================================
# BASE IMAGE 
#Ubuntu-based minimal Linux filesystem
#Python runtime
#system libraries
# ============================================================
# ============================================================
# BASE IMAGE
# ============================================================
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN pip install dist/*.whl || true
CMD ["python", "main.py"]

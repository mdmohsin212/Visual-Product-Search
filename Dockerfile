# FROM python:3.12-slim

# WORKDIR /app

# COPY requirements.txt .
# RUN pip install --upgrade pip
# RUN pip install -r requirements.txt

# COPY . .

# EXPOSE 10000

# CMD ["python", "app.py"]

FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git wget curl build-essential \
 && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python", "app.py"]
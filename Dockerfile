FROM python:3.13-slim-trixie
ENV PIP_ROOT_USER_ACTION=ignore
WORKDIR /app
RUN apt-get update -y && apt-get install awscli -y && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python3", "app.py"]
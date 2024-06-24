# app/Dockerfile

FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt ./requirements.txt

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Remove any existing files in the working directory
RUN rm -rf /app/*

# Other Dockerfile instructions https://heroku:<your_heroku_api_key>@git.heroku.com/bratsmlpredictor.git .

#RUN git clone https://heroku:HRKU-fac92678-5638-4a09-b87b-2490756fe01f@git.heroku.com/bratsmlpredictor.git .

COPY . .

WORKDIR /app

RUN pip3 install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "homepage.py", "--server.port=8501", "--server.address=0.0.0.0"]
# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app


RUN apt-get update && DEBIAN_FRONTEND=“noninteractive” apt-get install -y --no-install-recommends \
       nginx \
       ca-certificates \
       apache2-utils \
       certbot \
       python3-certbot-nginx \
       sudo \
       cifs-utils \
       && \
    rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get -y install cron


# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader punkt

# Make port 8501 available to the world outside this container
EXPOSE 8501
EXPOSE 80
EXPOSE 443

# Define environment variable
ENV NAME World

COPY nginx.conf /etc/nginx/nginx.conf
USER root

RUN chmod +x ./startup.sh
# Run app.py when the container launches
CMD ["./startup.sh"]

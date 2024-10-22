# Description: Dockerfile for the model microservice

# Use the official Python image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
# RUN pip install -r requirements.txt

# Install flask
RUN pip install flask

# Copy the current directory contents into the container at /app
COPY . .

# Expose the port 5000
EXPOSE 5000

# Run app.py when the container launches
CMD ["python", "app.py"]


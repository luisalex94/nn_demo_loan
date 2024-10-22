# Description: Dockerfile for the model microservice

# Use the official Python image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY requirements.txt requirements.txt
COPY app.py app.py

# Copy the model and preprocessing script
COPY model.pth model.pth
COPY preprocessor.pkl preprocessor.pkl

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install flask
RUN pip install flask

# Copy the current directory contents into the container at /app
COPY . .

# Expose the port 5000
EXPOSE 5000

# Run app.py when the container launches
CMD ["python", "app.py"]


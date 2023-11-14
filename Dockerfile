# Use the official Python image as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install any necessary dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Git and wget
RUN apt-get update && apt-get install -y git wget

# Clone the repository (without git lfs)
RUN git clone https://github.com/Ahmed-Essaam/floor-plan-analysis

# Copy the rest of your application code into the container
COPY . .

# Expose the port your Flask app will run on (default is 5000)
EXPOSE 8080

# Define the command to run your Flask app
CMD ["python", "handler.py"]

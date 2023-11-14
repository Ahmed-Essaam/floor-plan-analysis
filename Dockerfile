# Use the official Python image as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install any necessary dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Git LFS to fetch the large model file
RUN apt-get update && apt-get install -y git-lfs
RUN git lfs install

# Clone the Git LFS repository containing the large model file
RUN git lfs clone https://github.com/Ahmed-Essaam/floor-plan-analysis

# Copy the first large model file from the cloned repository to your app directory
RUN cp https://github.com/Ahmed-Essaam/floor-plan-analysis/blob/main/floortrans/models/model_1427.pth .

# Copy the large model file from the cloned repository to your app directory
RUN cp https://github.com/Ahmed-Essaam/floor-plan-analysis/blob/main/floortrans/models/model_best_val_loss_var.pkl .

# Copy the rest of your application code into the container
COPY . .

# Expose the port your Flask app will run on (default is 5000)
EXPOSE 8080

# Define the command to run your Flask app
CMD ["python", "handler.py"]

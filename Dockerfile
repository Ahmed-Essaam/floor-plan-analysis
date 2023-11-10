# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.9-slim

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# Set the GitHub token as a build argument
ARG GITHUB_TOKEN
ENV GITHUB_TOKEN=$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

# Copy the .gitattributes file first to ensure proper LFS handling
COPY .gitattributes ./

# Install Git LFS
RUN apt-get update && apt-get install -y git-lfs

# Clone the repository and fetch LFS files using the GitHub token
RUN git config --global credential.helper store && \
    echo "https://Ahmed-Essaam:${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}@github.com" > ~/.git-credentials && \
    git config --global user.email "bimquoteahmed@gmail.com" && \
    git config --global user.name "Ahmed-Essaam" && \
    git lfs install && \
    git clone https://<username>@github.com/Ahmed-Essaam/floor-plan-analysis.git && \
    cd floor-plan-analysis && \
    git lfs fetch

# Set the working directory
WORKDIR /app

# Copy local code to the container image.
COPY . .

# Install production dependencies.
RUN pip install -r requirements.txt

# Run the web service on container startup.
CMD exec gunicorn --bind :$PORT --workers 8 --threads 8 --timeout 0 handler:app

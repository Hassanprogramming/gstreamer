# Use a base image with GStreamer support
FROM ubuntu:latest

# Install GStreamer and dependencies
RUN apt update && apt install -y \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    python3 python3-pip

# Set the working directory
WORKDIR /app

# Copy project files
COPY . .

# Install Python dependencies if needed
RUN pip install -r requirements.txt || echo "No Python dependencies found"

# Expose ports if needed
EXPOSE 5000

# Define the entry point (modify as per your project)
CMD ["python3", "main.py"]

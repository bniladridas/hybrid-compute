# Dockerfile for hybrid-compute (local CPU components)
FROM ubuntu:24.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    clang \
    libopencv-dev \
    python3 \
    python3-pip \
    python3-opencv \
    imagemagick \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy source
COPY . .

# Install Python dependencies
RUN pip3 install --break-system-packages -r requirements.txt

# Build the project
RUN mkdir build && cd build && cmake .. -DWITH_OPENCV=ON && make

# Default command
CMD ["python3", "scripts/e2e.py"]

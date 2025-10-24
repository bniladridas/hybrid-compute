# Dockerfile for hybrid-compute (local CPU components)
FROM ubuntu:24.04 AS builder

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install build dependencies
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

WORKDIR /app

# Copy source
COPY . .

# Install Python dependencies
RUN pip3 install --break-system-packages --no-cache-dir -r requirements.txt

# Build the project
RUN mkdir build && cd build && cmake .. -DWITH_OPENCV=ON && make

# Runtime stage
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    python3 \
    python3-opencv \
    imagemagick \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy built binaries from builder
COPY --from=builder /app/build /app/build

# Default command
CMD ["python3", "scripts/e2e.py"]

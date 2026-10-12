FROM python:3.11-slim

# ffmpeg is required for video decoding
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir numpy scipy opencv-python matplotlib

COPY funautoscripter.py .
COPY src/ src/

ENTRYPOINT ["python", "funautoscripter.py"]

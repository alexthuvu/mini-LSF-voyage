# Use python 3.10 
FROM python:3.10-slim

# Add OpenCV dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

# Setup non-root user (required for Hugging Face)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy your entire app
COPY --chown=user . /app

# Set port 
EXPOSE 7860

# Start your Flask app (make sure `app.py` is under /app)
CMD ["python", "app.py"]
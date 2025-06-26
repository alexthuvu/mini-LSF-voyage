# Use python 3.10 
FROM python:3.10-slim-bullseye

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy your entire app
COPY . .

# Set port 
EXPOSE 7860

# Start your Flask app (make sure `app.py` is under /app)
CMD ["python", "app/app.py"]
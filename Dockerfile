# Use python 3.10 
FROM python:3.10-slim-bullseye

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

# Set working directory
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY . .

# Set port 
EXPOSE 7860

CMD ["python", "app/app.py"]

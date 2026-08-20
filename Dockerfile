FROM python:3.14-slim

# Set workdir
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Set default command (can be overridden)
CMD ["python", "run_experiment_matrix.py"]

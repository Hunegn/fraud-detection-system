# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5000 so that the container can communicate externally
EXPOSE 5000

# Set environment variable for Flask (optional)
ENV FLASK_APP=serve_model.py

# Run the Flask application when the container launches
CMD ["python", "api/serve_model.py"]

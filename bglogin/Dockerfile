# Use an official Python runtime as a parent image
'''FROM python:3.11.2-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed dependencies specified in requirements.txt
RUN pip install --no-cache-dir Flask numpy pandas scikit-learn joblib

# Make port 5000 available to the world outside this container
EXPOSE 5001

# Define environment variable
ENV FLASK_APP app.py

# Run app.py when the container launches
CMD ["python", "app.py",  "0.0.0.0:5001"]'''
# Use an official Python runtime as a parent image
FROM python:3.11.5-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed dependencies specified in requirements.txt
#RUN pip install --no-cache-dir Flask numpy pandas scikit-learn joblib
RUN pip install -r requirements.txt
# Make port 5000 available to the world outside this container
EXPOSE 5001

# Define environment variable
#ENV FLASK_APP app.py

# Run app.py when the container launches
CMD ["python", "app.py", "0.0.0.0:5001"]

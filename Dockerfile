FROM python:3.6

# Install necessary packages
RUN apt-get update
RUN apt-get install -y python-opencv

# install python packages   
RUN pip install pillow flask-socketio eventlet tensorflow keras numpy

# cleanup 
RUN apt-get clean
  
# Copy the current directory contents into the container at /app
RUN mkdir /app
COPY bot/ /app
  
# Set the working directory to /app
WORKDIR /app

# Run sample_bot.py when the container launches, you should replace it with your program
ENTRYPOINT ["python3", "bot.py"]

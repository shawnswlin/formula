FROM centos:7
  
# Install necessary packages
RUN yum install -y epel-release
RUN yum install -y python-pip
RUN yum install -y opencv-python
  
RUN pip install pillow flask-socketio eventlet
  
# Copy the current directory contents into the container at /app
RUN mkdir /app
COPY bot/ /app
  
# Set the working directory to /app
WORKDIR /app
# Run sample_bot.py when the container launches, you should replace it with your program
ENTRYPOINT ["python", "bot.py"]

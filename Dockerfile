# get base image and right Python version
FROM ubuntu:18.04
FROM python:3.6.6

# install python and pip
RUN apt-get update && apt-get install -y python-pip

# get dependencies
ADD requirements.txt /
RUN pip install -r requirements.txt

# copy files required for the app to run
COPY altair_pipeline.py /usr/src/app/
RUN mkdir /usr/src/app/modules
# copy Python modules
COPY modules/*py /usr/src/app/modules/
# copy config file
COPY modules/*ini /usr/src/app/modules/

# run the application
CMD python3 /usr/src/app/altair_pipeline.py
#CMD python3 /usr/src/app/test_docker_script.py

# our base image# our base image
FROM ubuntu:18.04
#ARG PYTHON_VERSION=3.6.6
FROM python:3.6.6

# install python and pip
#RUN apt-get install python-pip
RUN apt-get update && apt-get install -y python-pip

RUN python --version

# get dependencies
ADD requirements.txt /
#RUN curl https://bootstrap.pypa.io/get-pip.py | python 
RUN pip install -r requirements.txt

# copy files required for the app to run
COPY altair_pipeline.py /usr/src/app/

# run the application
CMD python3 /usr/src/app/altair_pipeline.py

Bootstrap: docker
From: python:3.6.6
#From: mwanakijiji/fizeau_altair_pipeline:latest

%files
requirements.txt /
altair_pipeline.py /usr/src/app/
modules/*py /usr/src/app/modules/
modules/*ini /usr/src/app/modules/

%post
# get base image and right Python version

# install python and pip
apt-get update && apt-get install -y python-pip

# get dependencies
pip install -r requirements.txt

# copy files required for the app to run
mkdir /usr/src/app/modules
# copy Python modules
# copy config file

# run the application
#CMD python3 /usr/src/app/test_docker_script.py
%runscript
exec /bin/bash python3 /usr/src/app/altair_pipeline.py "$@"
%startscript
exec /bin/bash python3 /usr/src/app/altair_pipeline.py "$@"

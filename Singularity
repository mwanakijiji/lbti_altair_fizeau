Bootstrap: docker
From: python:3.6.6

# copy files required for the app to run
mkdir -p /usr/src/app/modules

%files
  requirements.txt /
  altair_pipeline.py /usr/src/app/
  # copy Python modules
  modules/*py /usr/src/app/modules/
  # copy config file
  modules/*ini /usr/src/app/modules/

%post
  # install pip
  apt-get install -y python-pip
  # get dependencies
  pip install -r requirements.txt

# run the application (step not necessary if the verbose version
# of the command is in the PBS file)
#CMD python3 /usr/src/app/test_docker_script.py
%runscript
  echo "Runscript; Python version is"
  python --version
  #exec /bin/bash python3 /usr/src/app/altair_pipeline.py "$@"
%startscript
  echo "Startscript"
  #exec /bin/bash python3 /usr/src/app/altair_pipeline.py "$@"

Bootstrap: docker
From: python:3.6.6
#From: mwanakijiji/fizeau_altair_pipeline:latest

# copy files required for the app to run
mkdir -p /usr/src/app/modules

%files
  requirements.txt /
  altair_pipeline.py /usr/src/app/
  # copy Python modules
  ##modules/*py /usr/src/app/modules/
  # copy config file
  ##modules/*ini /usr/src/app/modules/

%post
  # install pip
  apt-get install -y python-pip
  # get dependencies
  pip install -r requirements.txt

  # UA HPC specific: make directories for mount points
  mkdir -p /extra
  mkdir -p /xdisk

# run the application
#CMD python3 /usr/src/app/test_docker_script.py
%runscript
  python --version
  #exec /bin/bash python3 /usr/src/app/altair_pipeline.py "$@"
%startscript
  echo "hello"
  #exec /bin/bash python3 /usr/src/app/altair_pipeline.py "$@"

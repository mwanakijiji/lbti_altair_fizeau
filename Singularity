Bootstrap: docker
From: python:3.6.6

# copy files required for the app to run
%setup
  mkdir -p ${SINGULARITY_ROOTFS}/modules
  mkdir -p ${SINGULARITY_ROOTFS}/vol_c

%files
  requirements.txt /
  altair_pipeline.py /
  # copy Python modules
  modules/*py /modules/
  # copy config file
  modules/*ini /modules/

%post
  # install pip
  apt-get update
  apt-get install -y python3-pip
  pip install -U pip
  # get dependencies
  pip install -r requirements.txt

# run the application (step not necessary if the verbose version
# of the command is in the PBS file)

%runscript
  echo "Runscript; Python version is"
  python --version
  #exec /bin/bash python3 /usr/src/app/altair_pipeline.py "$@"
%startscript
  echo "Startscript"
  #exec /bin/bash python3 /usr/src/app/altair_pipeline.py "$@"

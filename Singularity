Bootstrap: docker
From: python:3.6.6

# copy files required for the app to run
%setup
  mkdir -p ${SINGULARITY_ROOTFS}/usr/src/app/modules

%files
  requirements.txt /
  altair_pipeline.py ${SINGULARITY_ROOTFS}/usr/src/app/
  # copy Python modules
  modules/*py ${SINGULARITY_ROOTFS}/usr/src/app/modules/
  # copy config file
  modules/*ini ${SINGULARITY_ROOTFS}/usr/src/app/modules/

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

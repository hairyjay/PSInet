Installation steps to get ESPnet installed on my laptop using conda.

1. #create a conda environment
install miniconda for your ubuntu installation
init an environment and call it whatever you want ($env from here on)


retrieve epsnet from the github
install kaldi using the pre-built binaris
cd <espnet root folder>/ci/
bash install_kaldi.sh

#then follow the instructions from the installation instructions
https://espnet.github.io/espnet/installation.html

create a symlink to the kaldi
cd <espnet root folder>/tools
ln -s ../ci/tools/kaldi .

#now to set it up with anaconda

cd <espnet-root>/tools
CONDA_TOOLS_DIR=$(dirname ${CONDA_EXE})/..
./setup_anaconda.sh ${CONDA_TOOLS_DIR} [conda-env-name] [python-version]

where conda-env-name is $env like we mentioned earlier

then run make
cd <espnet-root>/tools
make

Run the below to check the installation
cd <espnet-root>/tools
. ./activate_python.sh; python3 check_install.py

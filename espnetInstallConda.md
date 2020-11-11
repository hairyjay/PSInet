Installation steps to get ESPnet installed on my laptop using conda.

1. create a conda environment
   install miniconda for your ubuntu installation (technically espnet does an install of miniconda for you but Rav is working off of a preinitialized miniconda) 
   init an environment and call it whatever you want ($env from here on)
   also init the environment with python 3.7

2. retrieve epsnet from the github (also available in this github)
   install kaldi using the pre-built binaries

```
cd <espnet root folder>/ci/
bash install_kaldi.sh
```

3. Then follow the instructions from the installation instructions For the most part replicated here https://espnet.github.io/espnet/installation.html

3a. create a symlink to the kaldi
```
cd <espnet root folder>/tools
ln -s ../ci/tools/kaldi .
```

3a. Set up the Cuda environment.
```
cd <espnet root>/tools
. ./setup_cuda_env.sh <cuda-root>
```
// Note for Rav and his Serve Setup the path to cuda root is ~/Classwork/Project/cuda because symlink b.s. had to happen.


3b. set up with the conda install we have
```
cd <espnet-root>/tools
CONDA_TOOLS_DIR=$(dirname ${CONDA_EXE})/..
./setup_anaconda.sh ${CONDA_TOOLS_DIR} $env [python-version]
```
where $env is the environment name  like we mentioned earlier

3ba. Debug
If you get an error that JAVA_HOME is an unbound variable then that means that Java is not installed on your system and can be done inside anaconda with 
    `conda install -c anaconda openjdk`
    

then run make
cd <espnet-root>/tools
make

Run the below to check the installation
cd <espnet-root>/tools
. ./activate_python.sh; python3 check_install.py

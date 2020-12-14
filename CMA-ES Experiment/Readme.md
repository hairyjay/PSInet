
In this sub directory we have all of the code used for running the CMA-ES experiment. To run the search experiment make sure that this espnet is properly installed, and then make sure pycma is installed either through conda or pip (I forget how I installed it) and then cd into the cma-es folder and run the cma_launcher.py script. it will do 20 runs total in lockstep parallel over 2 gpus (0 and 1).

To replicate the results in the paper, head over to and run the recipes in the espnet/egs2/wsj-exp* folders.

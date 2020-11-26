'''
Param name     Target Range Description                             Default
--num-mel-bins 70-90       Number of triangular mel-frequency bin  25/80
--frame-length 20 - 30      Frame length in milliseconds            25
--frame-shift  5-15         Frame shift in milliseconds             10
--min-f0       50 - 65      min. F0 to search for (Hz)              50
--max-f0       385-400      max. F0 to search for (Hz)              400
'''

import cma
from cma.fitness_transformations import EvalParallel2

import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import concurrent.futures

BOUNDS = [np.array([60,  20,  5, 50,  385]),
          np.array([100, 30, 15, 65, 400])]

init = np.array([80,25,10,50,400])

def launch_run(hyperparameters, cuda_dev=-1):
    """
        TODO: write training run code for ASR
            - Unroll Hyperparameter array
            - Generate necessary scripts
            - Run bash scripts
            - Read and evaluate WER/CER
            - Return to CMA-ES
        TODO: integrate with ESPnet and feature extraction model training procedures

        Generate a training run of ASR system with selected hyperparameters
        :param hyperparameters: 1D array of hyperparameter values
        :return: evaluation metric
    """


    if(cuda_dev == -1):
        raise Exception("incorrect cuda device")

    basepath = "../espnet/egs2/wsj%s/asr1/"


    hyperparameters = hyperparameters.astype(int)
    num_mel_bins, frame_length, frame_shift, min_f0, max_f0 = hyperparameters
    def f(x):
        return np.sum(x**2)
    print("inside Function", hyperparameters)

    return f(hyperparameters)

options = cma.CMAOptions()
options.set('bounds', BOUNDS)
options.set('popsize', 4)
options.set('integer_variable', list(range(len(init))))
options.set('maxfevals', 32)
#options.set('CMA_cmean', 4)
options.set('CMA_stds', [2]*len(BOUNDS[0]))

es = cma.CMAEvolutionStrategy(init, 2, options)
f = cma.s.ft.IntegerMixedFunction(launch_run, np.arange(5))


#for i in range(5):
#    print(es.ask())

while not es.stop():
    attempts = es.ask()

    #evaluate the 2 attempts by multi-threading the launch_run function
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future1 = executor.submit(launch_run, attempts[0], 3)
        future2 = executor.submit(launch_run, attempts[1], 4)

        toTell = [future1.result(), future2.result()]

        future1 = executor.submit(launch_run, attempts[2], 3)
        future2 = executor.submit(launch_run, attempts[3], 4)

        toTell += [future1.result(), future2.result()]

        print(toTell)

    es.tell(attempts, toTell)
    es.logger.add(es)
    es.disp()








#es.optimize(f)
#with EvalParallel2(f, number_of_processes=2) as evalP: #SET number_of_processes
#    while not es.stop():
#        X = es.ask()
#        es.tell(X, evalP(X)) #CHANGE eval() to TRAINING WRAPPER
print("Buffer\n\n")
es.result_pretty()

es.logger.save()
print(np.sum(BOUNDS[0]**2))

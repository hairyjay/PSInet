'''
Param name     Target Range Description                             Default
--num-mel-bins 20-160       Number of triangular mel-frequency bin  25/80
--frame-length 15-40        Frame length in milliseconds            25
--frame-shift  5-15         Frame shift in milliseconds             10
'''

#average the wer on the two datasets
import cma
from cma.fitness_transformations import EvalParallel2

import subprocess

import numpy as np

#import torch
#import torchvision
#import torch.nn as nn
#import torch.nn.functional as F
#import torch.optim as optim

import concurrent.futures

BOUNDS = [np.array([4,  15,  5,]),
          np.array([32, 40, 15,])]

init = np.array([5,25,10])

def launch_run(hyperparameters, cuda_dev=-1, dup = ""):
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

    print("RUNNING:" + dup + "\n\n")

    if(cuda_dev == -1):
        raise Exception("incorrect cuda device")


    basepath = "../espnet/egs2/wsj%s/asr1/"%(dup)
    #basepath = "../espnet/egs2/wsj%s/asr1/"

    #command
    expType = 'fbank_pitch'
    cmd = "CUDA_VISIBLE_DEVICES=%s ./run.sh --ngpu 1 --feats_type %s" % (cuda_dev, expType)

    hyperparameters = hyperparameters.astype(int)
    num_mel_bins, frame_length, frame_shift = hyperparameters
    num_mel_bins *= 5
    
    
    #modify the files at the basepath
    fbank = "\n".join(["--sample-frequency=16000", "--num-mel-bins=%s"%(num_mel_bins)])

    with open(basepath + "conf/fbank.conf", 'w') as f:
        f.write(fbank)

    pitch = "\n".join(["--sample-frequency=16000", "--frame-length=%s"%(frame_length), "--frame-shift=%s" %(frame_shift)])
    with open(basepath + "conf/pitch.conf", 'w') as f:
        f.write(pitch)

    #write a bash file to run

    fh = open("run%s.sh" %(dup), "w")
    fh.write("export JAVA_LD_LIBRARY_PATH='/afs/cs.pitt.edu/usr0/ras306/miniconda3/envs/psinet/jre/lib/amd64/server'\n")
    fh.write("export JAVA_HOME='/afs/cs.pitt.edu/usr0/ras306/miniconda3/envs/psinet'\n")
    fh.write("cd %s\n" % (basepath))
    fh.write(cmd + "\n")

    fh.close()
    
    #run the file and wait to finish

    results = subprocess.run(['bash', 'run%s.sh'%(dup)], stderr=subprocess.PIPE, text=True)

    #read in the results.
    #print(results.stderr)
    #print("\n\n")

    if('failed' in results.stderr):
        retval = 1000
        print(results.stderr)
        print("Failed a run")

    else:
        #assume success and read in the result
        exp_name = 'asr_train_asr_transformer_%s_char' %(expType)
        with open(basepath+"exp/%s/RESULTS.md" %(exp_name), 'r') as f:
            data = []
            retval = -1
            content = f.readlines()
            for i in range(len(content)):
                if("WER" in content[i]):
                    v1 = float(content[i+4].split("|")[-3])
                    v2 = float(content[i+5].split("|")[-3])

        retval = (v1 + v2)/2
        print(retval)

    #delete the dump 
    
    return retval

#print(launch_run(np.array([81,27,9]), 7))

best_sol = cma.optimization_tools.BestSolution(x = [5,25,10], f = 13.5)

options = cma.CMAOptions()
options.set('bounds', BOUNDS)
options.set('popsize', 4)
options.set('integer_variable', list(range(len(init))))
#options.set('maxfevals', 32)
#options.set('maxfevals', 8)
#options.set('CMA_cmean', 4)
options.set('CMA_stds', [4]*len(BOUNDS[0]))
options.set('verb_append', best_sol.evalsall)

es = cma.CMAEvolutionStrategy(init, 4, options)
f = cma.s.ft.IntegerMixedFunction(launch_run, np.arange(5))

detailLog = open("allRuns.log", 'w')

maxFevals = 32
curFevals = 0

#for i in range(5):
#    print(es.ask())

while curFevals != maxFevals:
    attempts = es.ask()

    #evaluate the 2 attempts by multi-threading the launch_run function
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future1 = executor.submit(launch_run, attempts[0], 6)
        future2 = executor.submit(launch_run, attempts[1], 7, "-dup")
        
        if(future1.result() != 1000):
            curFevals += 1
        if(future2.result() != 1000):
            curFevals += 1

        toTell = [future1.result(), future2.result()]
        future1 = executor.submit(launch_run, attempts[2], 6)
        future2 = executor.submit(launch_run, attempts[3], 7, "-dup")
        
        if(future1.result() != 1000):
            curFevals += 1
        if(future2.result() != 1000):
            curFevals += 1

        toTell += [future1.result(), future2.result()]

        for i in range(4):
            msg = str(toTell[i]) +  "\t" +str(attempts[i])
            detailLog.write(msg + "\n")

        detailLog.flush()
        

        print(toTell)

    es.tell(attempts, toTell)
    es.logger.add(es)
    es.disp()
detailLog.close()

#es.optimize(f)
#with EvalParallel2(f, number_of_processes=2) as evalP: #SET number_of_processes
#    while not es.stop():
#        X = es.ask()
#        es.tell(X, evalP(X)) #CHANGE eval() to TRAINING WRAPPER
print("Buffer\n\n")
es.result_pretty()

es.logger.save()

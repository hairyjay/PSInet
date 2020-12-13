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

modelConf='''batch_type: folded
batch_size: 32
accum_grad: 8
max_epoch: 100
patience: none
# The initialization method for model parameters
init: xavier_uniform
best_model_criterion:
-   - valid
    - acc
    - max
keep_nbest_models: 10

#location doesn't matter but let's keep it in order for ease of reading
frontend: default
frontend_conf:
    n_mels: %s
    n_fft: %s
    hop_length: %s

encoder: transformer
encoder_conf:
    output_size: 256
    attention_heads: 4
    linear_units: 2048
    num_blocks: 12
    dropout_rate: 0.1
    positional_dropout_rate: 0.1
    attention_dropout_rate: 0.0
    input_layer: conv2d
    normalize_before: true

decoder: transformer
decoder_conf:
    attention_heads: 4
    linear_units: 2048
    num_blocks: 6
    dropout_rate: 0.1
    positional_dropout_rate: 0.1
    self_attention_dropout_rate: 0.0
    src_attention_dropout_rate: 0.0

model_conf:
    ctc_weight: 0.3
    lsm_weight: 0.1
    length_normalized_loss: false

optim: adam
optim_conf:
    lr: 0.005
scheduler: warmuplr
scheduler_conf:
    warmup_steps: 30000
'''

# modelConf='''batch_type: folded
# batch_size: 64
# accum_grad: 2
# max_epoch: 200
# patience: none
# # The initialization method for model parameters
# init: xavier_uniform
# best_model_criterion:
# -   - valid
#     - acc
#     - max
# keep_nbest_models: 10

# #location doesn't matter but let's keep it in order for ease of reading
# frontend: default
# frontend_conf:
#     n_mels: %s
#     win_length: %s
#     hop_length: %s


# encoder: transformer
# encoder_conf:
#     output_size: 256
#     attention_heads: 4
#     linear_units: 2048
#     num_blocks: 12
#     dropout_rate: 0.1
#     positional_dropout_rate: 0.1
#     attention_dropout_rate: 0.0
#     input_layer: conv2d
#     normalize_before: true

# decoder: transformer
# decoder_conf:
#     attention_heads: 4
#     linear_units: 2048
#     num_blocks: 6
#     dropout_rate: 0.1
#     positional_dropout_rate: 0.1
#     self_attention_dropout_rate: 0.0
#     src_attention_dropout_rate: 0.0

# model_conf:
#     ctc_weight: 0.3
#     lsm_weight: 0.1
#     length_normalized_loss: false

# optim: adam
# optim_conf:
#     lr: 0.005
# scheduler: warmuplr
# scheduler_conf:
#     warmup_steps: 20000
# '''

import concurrent.futures

BOUNDS = [np.array([0, 0, 0]),
          np.array([1, 1,.5])]

init = np.array([.428, .333, .28])

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

    # hyperparameters = hyperparameters.astype(int)
    num_mel_bins, frame_length, frame_shift = hyperparameters
    num_mel_bins = int(140*num_mel_bins + 20)
    frame_length = int(768*frame_length + 256)
    frame_shift = int(100*frame_shift + 100)

    print("RUNNING: " + dup + "\nHYPERPARAMETER VALUES: num_mel_bins = %s, n_ftt = %s, frame_shift = %s\n\n\n"%(num_mel_bins, frame_length, frame_shift))


    #return np.sum(hyperparameters**2)

    if(cuda_dev == -1):
        raise Exception("incorrect cuda device")


    basepath = "../espnet/egs2/wsj%s/asr1/"%(dup)
    #basepath = "../espnet/egs2/an4%s/asr1/"%(dup)

    #command
    expType = 'raw'
    cmd = "CUDA_VISIBLE_DEVICES=%s ./run.sh --ngpu 1 --feats_type %s" % (cuda_dev, expType)


    #save the parameters
    conf = modelConf % (num_mel_bins, frame_length, frame_shift)
    with open(basepath + "conf/train_asr_transformer.yaml", 'w') as f:
        f.write(conf)


    #modify the files at the basepath
    # fbank = "\n".join(["--sample-frequency=16000", "--num-mel-bins=%s"%(num_mel_bins)])

    # with open(basepath + "conf/fbank.conf", 'w') as f:
    #     f.write(fbank)

    # pitch = "\n".join(["--sample-frequency=16000", "--frame-length=%s"%(frame_length), "--frame-shift=%s" %(frame_shift)])
    # with open(basepath + "conf/pitch.conf", 'w') as f:
    #     f.write(pitch)C

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
        #exp_name = 'asr_train_%s_bpe30' %(expType)
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

    #delete the dump and exp folder
    #delRes = subprocess.run(["rm", "-r", basepath+"dump/"])
    delRes = subprocess.run(["rm", "-r", basepath+"exp/"])

    return retval

#print(launch_run(np.array([81,27,9]), 7))

def try_values(es, dev=-1, dup = ""):
    result = 1000
    attempt = None
    i = 0
    while result >= 1000 and i < 50:
        attempt = es.ask(number=1)[0]
        result = launch_run(attempt, dev, dup)
        i += 1
    return attempt, result

options = cma.CMAOptions()
options.set('bounds', BOUNDS)
options.set('popsize', 4)
options.set('integer_variable', list(range(len(init))))
#options.set('maxfevals', 32)
#options.set('maxfevals', 8)
#options.set('CMA_cmean', 4)
options.set('CMA_stds', [.2,.2,.08])

es = cma.CMAEvolutionStrategy(init, 4, options)
f = cma.s.ft.IntegerMixedFunction(launch_run, np.arange(5))

detailLog = open("allRuns.log", 'w')

maxFevals = 20
curFevals = 0

#for i in range(5):
#    print(es.ask())

while curFevals != maxFevals:
    attempts = es.ask()

    #evaluate the 2 attempts by multi-threading the launch_run function
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # future1 = executor.submit(try_values, es, 0)
        # future2 = executor.submit(try_values, es, 1, "-dup")
        # future3 = executor.submit(try_values, es, 2, "-dup2")
        # attempt1 = es.ask(number=1)[0]
        # attempt2 = es.ask(number=1)[0]
        # attempt3 = es.ask(number=1)[0]
        future1 = executor.submit(launch_run, attempts[0], 0)
        future2 = executor.submit(launch_run, attempts[1], 1, "-dup")
        # future3 = executor.submit(launch_run, attempt1, 2, "-dup2")
        #future4 = executor.submit(try_values, es, 3, "-dup3")
        if(future1.result() != 1000):
            curFevals += 1
        if(future2.result() != 1000):
            curFevals += 1
        # if(future3.result() != 1000):
        #     curFevals += 1
        #if(future4.result() != 1000):
         #   curFevals += 1
        # attempts = [future1.result()[0], future2.result()[0]]
        # toTell = [future1.result()[1], future2.result()[1]]

        # attempts = [future1.result()[0],
        #             future2.result()[0],
        #             future3.result()[0],
        #  #           future4.result()[0],
        # ]

        # attempt = [attempt1,
        #            attempt2,
        # ]

        toTell =[future1.result(),
                 future2.result(),
                 #future3.result()[1],
        #         future4.result()[1],
        ]
        future1 = executor.submit(launch_run, attempts[2], 0)
        future2 = executor.submit(launch_run, attempts[3], 1, "-dup")
        if(future1.result() != 1000):
            curFevals += 1
        if(future2.result() != 1000):
            curFevals += 1
        # future1 = executor.submit(try_values, es, 0)
        # future2 = executor.submit(try_values, es, 1, "-dup")
        # attempts += [future1.result()[0],
        #             future2.result()[0],
        #             future3.result()[0],
        #  #           future4.result()[0],
        # ]
        toTell +=[future1.result(),
                 future2.result(),
                 #future3.result()[1],
        #         future4.result()[1],
        ]
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

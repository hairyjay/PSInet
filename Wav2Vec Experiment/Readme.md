# Wav2Vec Experiment

In this folder we handle the wav2vec experiments

## Wav2Vec Fine-tune Task
We followed instructions in fairseq, which provides us ways to fine-tune on Librispeech Dataset. (see fairseq/examples/wav2vec/)

### Data Preparation
We prepared WSJ in the same way, you can find detailed implementation in `Wav2Vec Experiment/wav2vec_scripts/fine-tune-wsj-data_prep.ipynb`.
First remember to install fairseq:
```
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
```

1. Since WSJ dataset contains sphere files, which can not be directly used. We first used sph2pipe to convert them to wav files.
```
./sph2pipe -f rif {/path/to/sph} {wav_file_name}
```
2. Built wav2vec manifest, which is a tsv file that contains wav paths. train_si284 was used to train and test\_dev93 was used to validate. That gave us two manifest files: `train.tsv` and `valid.tsv`
3. Prepared a letter dictionary `dict.ltr.txt`
4. created a parallel labels files corresponding line by line to the files in train.tsv, dev.tsv, this step gave us four files:
`train.ltr`, `train.wrd`, `valid.ltr`, `valid.wrd`
5. train_si284 has approximately 100 hours of audio data so we used examples/wav2vec/config/finetuning/base_100h.yaml as our configuration file.

Note: put all above files in the same folder as our /path/to/data

### Fine-tuning command
1. Download one pre-trained model from https://github.com/casssie-zhang/fairseq/tree/fb4bc34a7b2de1fbccdae65f099a321248430bc1/examples/wav2vec
2. Run the following command:
```
fairseq-hydra-train \
    distributed_training.distributed_port=$PORT \
    task.data=/path/to/data \
    model.w2v_path=/path/to/model.pt \
    --config-dir /path/to/fairseq-py/examples/wav2vec/config/finetuning \
    --config-name base_100h
```


## Integration of Wav2Vec and Espnet

To reproduce the wav2vec experiments 3 files need to be downloaded first and placed in a path that is easy to access.

They are located here: https://drive.google.com/drive/u/1/folders/1T6v1II5X387yZhyL78KnpEfpc3NbEN_R

The files are `dict.ltr.txt` `wav2vec_small.pt` `checkpoint_best.pt` and save them under some directory, let's say /models/ 

Then inside of the recipes a model path is specified and they should be updated with the path you downloaded the files on your local machine.

Furthermore some of the files in the fairseq directory need to be updated so that they correctly pull your files.

In particular they are 

fairseq/data/dictionary.py
fairseq/checkpoint_utils.py


once those are updated the wav2vec encoder should work properly inside of espnet.

The experiments to run inside of espnet are egs2/wsj-w2v-enc/ for to use just the pre-trained model

and egs2/wsj-w2v-enc2/ for the finetuned

There are also ones that contain a 50 in the path and those are just 50 epoch runs instead of full runs to get an idea of the performance.

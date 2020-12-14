

In this folder we handle the wav2vec experiments, to reproduce the wav2vec experiments 3 files need to be downloaded first and placed in a path that is easy to access.

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

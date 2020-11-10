# PSInet
11-785 Project

This is a Speech Representation project based on ESPnet

ESPnet is an end-to-end speech processing toolkit, mainly focuses on end-to-end speech recognition and end-to-end text-to-speech.
ESPnet uses [chainer](https://chainer.org/) and [pytorch](http://pytorch.org/) as a main deep learning engine,
and also follows [Kaldi](http://kaldi-asr.org/) style data processing, feature extraction/format, and recipes to provide a complete setup for speech recognition and other speech processing experiments.

#### We focus on a feature exploration task of trying to find new ways to process source signals in ASR systems.

### ASR: Automatic Speech Recognition
- **State-of-the-art performance** in several ASR benchmarks (comparable/superior to hybrid DNN/HMM and CTC)
- **Hybrid CTC/attention** based end-to-end ASR
  - Fast/accurate training with CTC/attention multitask training
  - CTC/attention joint decoding to boost monotonic alignment decoding
  - Encoder: VGG-like CNN + BiRNN (LSTM/GRU), sub-sampling BiRNN (LSTM/GRU) or Transformer
- Attention: Dot product, location-aware attention, variants of multihead
- Incorporate RNNLM/LSTMLM/TransformerLM/N-gram trained only with text data
- Batch GPU decoding
- **Transducer** based end-to-end ASR
  - Available: RNN-based encoder/decoder and Transformer-based encoder/decoder w/ customizable architecture.
  - Also support: mixed RNN/Transformer architecture, attention mechanism (RNN decoder), VGG2L (RNN/Transformer encoder), Conformer (Transformer encoder), TDNN (Transformer encoder), Causal Conv1d (Transformer decoder) and various decoding algorithms.
  > Please refer to the [tutorial page](https://espnet.github.io/espnet/tutorial.html#transducer) for complete documentation.
- CTC segmentation
- ASR examples for supporting endangered language documentation (Please refer to egs/puebla_nahuatl and egs/yoloxochitl_mixtec for details)



### ASR results
TO-DO: Update result of WSJ data

Below is the character error rate (CER) and word error rate (WER) of major ASR tasks using ESPnet.

Our task is to make improvements through feature processing.

| Task                   | CER (%) | WER (%) | Pretrained model|
| -----------            | :----:  | :----:  | :----:                                                                                                                                                                |
| Aishell dev/test            | 4.6/5.1    | N/A     | [link](https://github.com/espnet/espnet/blob/master/egs/aishell/asr1/RESULTS.md#conformer-kernel-size--15--specaugment--lm-weight--00-result) |
| **ESPnet2** Aishell dev/test            | 4.4/4.7    | N/A     | [link](https://github.com/espnet/espnet/tree/master/egs2/aishell/asr1#conformer--specaug--speed-perturbation-featsraw-n_fft512-hop_length128) |
| Common Voice dev/test       | 1.7/1.8     | 2.2/2.3     | [link](https://github.com/espnet/espnet/blob/master/egs/commonvoice/asr1/RESULTS.md#first-results-default-pytorch-transformer-setting-with-bpe-100-epochs-single-gpu) |
| CSJ eval1/eval2/eval3              | 5.7/3.8/4.2     | N/A     | [link](https://github.com/espnet/espnet/blob/master/egs/csj/asr1/RESULTS.md#pytorch-backend-transformer-without-any-hyperparameter-tuning)                            |
| **ESPnet2** CSJ eval1/eval2/eval3              | 4.5/3.3/3.6     | N/A     | [link](https://github.com/espnet/espnet/tree/master/egs2/csj/asr1#initial-conformer-results)                            |
| HKUST dev              | 23.5    | N/A     | [link](https://github.com/espnet/espnet/blob/master/egs/hkust/asr1/RESULTS.md#transformer-only-20-epochs)                                                             |
| Librispeech dev_clean/dev_other/test_clean/test_other  | N/A     | 1.9/4.9/2.1/4.9     | [link](https://github.com/espnet/espnet/blob/master/egs/librispeech/asr1/RESULTS.md#pytorch-large-conformer-with-specaug--speed-perturbation-8-gpus--transformer-lm-4-gpus)             |
| TEDLIUM2 dev/test           | N/A     | 8.6/7.2     | [link](https://github.com/espnet/espnet/blob/master/egs/tedlium2/asr1/RESULTS.md#conformer-large-model--specaug--speed-perturbation--rnnlm)   |
| TEDLIUM3 dev/test           | N/A     | 9.6/7.6     | [link](https://github.com/espnet/espnet/blob/master/egs/tedlium3/asr1/RESULTS.md)                   |
| WSJ dev93/eval92              | 3.2/2.1     | 7.0/4.7     | N/A |
|  **ESPnet2** WSJ dev93/eval92              | 2.7/1.8     | 6.6/4.6     | [link](https://github.com/espnet/espnet/tree/master/egs2/wsj/asr1#using-transformer-lm-asr-model-is-same-as-the-above-lm_weight12-ctc_weight03-beam_size20) |

Note that the performance of the CSJ, HKUST, and Librispeech tasks was significantly improved by using the wide network (#units = 1024) and large subword units if necessary reported by [RWTH](https://arxiv.org/pdf/1805.03294.pdf).

If you want to check the results of the other recipes, please check `egs/<name_of_recipe>/asr1/RESULTS.md`.

</div></details>


### ASR demo

<details><summary>expand</summary><div>

You can recognize speech in a WAV file using pretrained models.
Go to a recipe directory and run `utils/recog_wav.sh` as follows:
```sh
# go to recipe directory and source path of espnet tools
cd egs/tedlium2/asr1 && . ./path.sh
# let's recognize speech!
recog_wav.sh --models tedlium2.transformer.v1 example.wav
```
where `example.wav` is a WAV file to be recognized.
The sampling rate must be consistent with that of data used in training.

Available pretrained models in the demo script are listed as below.

| Model                                                                                            | Notes                                                      |
| :------                                                                                          | :------                                                    |
| [tedlium2.rnn.v1](https://drive.google.com/open?id=1UqIY6WJMZ4sxNxSugUqp3mrGb3j6h7xe)            | Streaming decoding based on CTC-based VAD                  |
| [tedlium2.rnn.v2](https://drive.google.com/open?id=1cac5Uc09lJrCYfWkLQsF8eapQcxZnYdf)            | Streaming decoding based on CTC-based VAD (batch decoding) |
| [tedlium2.transformer.v1](https://drive.google.com/open?id=1cVeSOYY1twOfL9Gns7Z3ZDnkrJqNwPow)    | Joint-CTC attention Transformer trained on Tedlium 2       |
| [tedlium3.transformer.v1](https://drive.google.com/open?id=1zcPglHAKILwVgfACoMWWERiyIquzSYuU)    | Joint-CTC attention Transformer trained on Tedlium 3       |
| [librispeech.transformer.v1](https://drive.google.com/open?id=1BtQvAnsFvVi-dp_qsaFP7n4A_5cwnlR6) | Joint-CTC attention Transformer trained on Librispeech     |
| [commonvoice.transformer.v1](https://drive.google.com/open?id=1tWccl6aYU67kbtkm8jv5H6xayqg1rzjh) | Joint-CTC attention Transformer trained on CommonVoice     |
| [csj.transformer.v1](https://drive.google.com/open?id=120nUQcSsKeY5dpyMWw_kI33ooMRGT2uF)         | Joint-CTC attention Transformer trained on CSJ             |
| [csj.rnn.v1](https://drive.google.com/open?id=1ALvD4nHan9VDJlYJwNurVr7H7OV0j2X9)                 | Joint-CTC attention VGGBLSTM trained on CSJ                |

</div></details>


## References

[1] Shinji Watanabe, Takaaki Hori, Shigeki Karita, Tomoki Hayashi, Jiro Nishitoba, Yuya Unno, Nelson Enrique Yalta Soplin, Jahn Heymann, Matthew Wiesner, Nanxin Chen, Adithya Renduchintala, and Tsubasa Ochiai, "ESPnet: End-to-End Speech Processing Toolkit," *Proc. Interspeech'18*, pp. 2207-2211 (2018)

[2] Suyoun Kim, Takaaki Hori, and Shinji Watanabe, "Joint CTC-attention based end-to-end speech recognition using multi-task learning," *Proc. ICASSP'17*, pp. 4835--4839 (2017)

[3] Shinji Watanabe, Takaaki Hori, Suyoun Kim, John R. Hershey and Tomoki Hayashi, "Hybrid CTC/Attention Architecture for End-to-End Speech Recognition," *IEEE Journal of Selected Topics in Signal Processing*, vol. 11, no. 8, pp. 1240-1253, Dec. 2017

## Citations

```
@inproceedings{watanabe2018espnet,
  author={Shinji Watanabe and Takaaki Hori and Shigeki Karita and Tomoki Hayashi and Jiro Nishitoba and Yuya Unno and Nelson {Enrique Yalta Soplin} and Jahn Heymann and Matthew Wiesner and Nanxin Chen and Adithya Renduchintala and Tsubasa Ochiai},
  title={{ESPnet}: End-to-End Speech Processing Toolkit},
  year={2018},
  booktitle={Proceedings of Interspeech},
  pages={2207--2211},
  doi={10.21437/Interspeech.2018-1456},
  url={http://dx.doi.org/10.21437/Interspeech.2018-1456}
}
@inproceedings{hayashi2020espnet,
  title={{Espnet-TTS}: Unified, reproducible, and integratable open source end-to-end text-to-speech toolkit},
  author={Hayashi, Tomoki and Yamamoto, Ryuichi and Inoue, Katsuki and Yoshimura, Takenori and Watanabe, Shinji and Toda, Tomoki and Takeda, Kazuya and Zhang, Yu and Tan, Xu},
  booktitle={Proceedings of IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={7654--7658},
  year={2020},
  organization={IEEE}
}
@inproceedings{inaguma-etal-2020-espnet,
    title = "{ESP}net-{ST}: All-in-One Speech Translation Toolkit",
    author = "Inaguma, Hirofumi  and
      Kiyono, Shun  and
      Duh, Kevin  and
      Karita, Shigeki  and
      Yalta, Nelson  and
      Hayashi, Tomoki  and
      Watanabe, Shinji",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics: System Demonstrations",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-demos.34",
    pages = "302--311",
}
```


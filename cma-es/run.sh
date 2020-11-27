export JAVA_LD_LIBRARY_PATH='/afs/cs.pitt.edu/usr0/ras306/miniconda3/envs/psinet/jre/lib/amd64/server'
export JAVA_HOME='/afs/cs.pitt.edu/usr0/ras306/miniconda3/envs/psinet'
cd ../espnet/egs2/wsj/asr1/
CUDA_VISIBLE_DEVICES=6 ./run.sh --ngpu 1 --feats_type fbank_pitch

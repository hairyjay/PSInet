export JAVA_LD_LIBRARY_PATH='/afs/cs.pitt.edu/usr0/ras306/miniconda3/envs/psinet/jre/lib/amd64/server'
export JAVA_HOME='/afs/cs.pitt.edu/usr0/ras306/miniconda3/envs/psinet'
cd ../espnet/egs2/wsj-dup1/asr1/
CUDA_VISIBLE_DEVICES=1 ./run.sh --ngpu 1 --feats_type raw

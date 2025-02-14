config=$1
modeldir=$2

CUDA_VISIBLE_DEVICES=0,1,2,3 python init.py -c $config -m $modeldir
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -c $config -m $modeldir

#!/usr/bin/env bash
CONFIG=$1
#GPUS=$2
DEVICE=$2
i=1
while ((1==1)); do
    tmp=`echo $DEVICE | cut -d "," -f $i`
    if [ "$tmp" == "" ]; then
        break;
    else
        ((i++))
    fi
done
GPUS=$[i-1]
export CUDA_VISIBLE_DEVICES=$DEVICE
echo $GPUS
echo $CUDA_VISIBLE_DEVICES
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch --no-validate ${@:3}

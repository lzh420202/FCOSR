#!/usr/bin/env bash


echo "800x800 benchmark"
echo "fcosr_mobilenetv2_fpn_40k_hrsc2016.py"
config_file=configs/fcosrbox/fcosr_mobilenetv2_fpn_40k_hrsc2016.py
checkpoint=work_dirs/HRSC/FCOSR_mobilenetv2_fpn_40k_hrsc2016_v1/iter_40000.pth
python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 tools/analysis_tools/benchmark.py $config_file $checkpoint --launcher pytorch --max-iter 444 --log-interval 20 --fuse-conv-bn

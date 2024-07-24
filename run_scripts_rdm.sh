#!/bin/bash

# 设置必要的环境变量（如果需要）
OUTPUT_DIR=./output
IMAGENET_DIR=../dataset/Sketchy/sketch/tx_000000000000
#MASTER_SERVER_ADDRESS=10.106.128.144
MASTER_SERVER_ADDRESS=localhost

# 运行 Python 脚本
python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --node_rank=0 \
main_rdm.py \
--config config/rdm/mocov3vitb_simplemlp_l12_w1536.yaml \
--batch_size 128 --input_size 256 \
--epochs 200 \
--blr 1e-6 --weight_decay 0.01 \
--output_dir ${OUTPUT_DIR} \
--data_path ${IMAGENET_DIR} \
--dist_url tcp://${MASTER_SERVER_ADDRESS}:20

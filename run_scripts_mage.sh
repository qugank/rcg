#!/bin/bash

# 设置必要的环境变量（如果需要）
RDM_CFG_PATH=config/rdm/mocov3vitb_simplemlp_l12_w1536.yaml
RDM_CKPT_PATH=pretrained_rdm/rdm-mocov3vitb.pth
OUTPUT_DIR=./output/mage
IMAGENET_DIR=../dataset/Sketchy/photo/tx_000000000000
#MASTER_SERVER_ADDRESS=10.106.128.144
MASTER_SERVER_ADDRESS=localhost

# 运行 Python 脚本
python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --node_rank=0 \
main_mage.py \
--pretrained_enc_arch mocov3_vit_base \
--pretrained_enc_path pretrained_enc_ckpts/mocov3/vitb.pth.tar --rep_drop_prob 0.1 \
--use_rep --rep_dim 256 --pretrained_enc_withproj --pretrained_enc_proj_dim 256 \
--pretrained_rdm_cfg ${RDM_CFG_PATH} --pretrained_rdm_ckpt ${RDM_CKPT_PATH} \
--rdm_steps 250 --eta 1.0 --temp 6.0 --num_iter 20 --num_images 50000 --cfg 0.0 \
--batch_size 64 --input_size 256 \
--model mage_vit_base_patch16 \
--mask_ratio_min 0.5 --mask_ratio_max 1.0 --mask_ratio_mu 0.75 --mask_ratio_std 0.25 \
--epochs 200 \
--warmup_epochs 10 \
--blr 1.5e-4 --weight_decay 0.05 \
--output_dir ${OUTPUT_DIR} \
--data_path ${IMAGENET_DIR} \
--dist_url tcp://${MASTER_SERVER_ADDRESS}:20

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=3721 basicsr/train.py \
    -opt options/train/train_ViCoW_1.0_0.5_freeze_g_nonorm_pretrain_512_kaggle.yml --auto_resume --launcher pytorch

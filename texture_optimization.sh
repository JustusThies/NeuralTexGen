#!/bin/bash
set -ex
# bash texture_optimization.sh


DATASETS_DIR=./datasets/
DATASET_MODE=uv
GPUID=0
#TARGET_ACTOR=room1
#TARGET_ACTOR=room1_lqlq
TARGET_ACTOR=room1_us


# texture
#TEX_DIM=1024
TEX_DIM=2048

# models
MODEL=RGBTextures

# model settings
TEXTURE_MODEL=StaticNeuralTexture   # only if NeutalTextures
RENDERER_TYPE=UNET_5_level          # only if NeutalTextures
NGF=32                              # only if NeutalTextures
TEX_FEATURES=8                      # only if NeutalTextures
TEX_FEATURES_INTERMEDIATE=16        # only if NeutalTextures

# optimizer parameters
#LR=0.01
LR=0.01

N_ITER=20 #25
N_ITER_LR_DECAY=40 #75 #25
BATCH_SIZE=1 # has to be 1!!

# loss
LOSS=VGG
#LOSS=L1

# save frequency
SAVE_FREQ=10

################################################################################
###############################    TRAINING     ################################
################################################################################

DATE_WITH_TIME=`date "+%Y%m%d-%H%M%S"`
NAME=$MODEL/$TARGET_ACTOR-TEX$TEX_DIM-$LOSS-$DATE_WITH_TIME-perEpoch-Adam
DISPLAY_NAME=${MODEL}/${TARGET_ACTOR}-TEX$TEX_DIM-${LOSS}-perEpoch-Adam

##
RESULT_DIR=./results/$TARGET_ACTOR/TEX$TEX_DIM-$LOSS-perEpoch-Adam-001/
mkdir -p $RESULT_DIR

# training
#--continue_train 
python train_acc.py --results_dir $RESULT_DIR --name $NAME --save_epoch_freq $SAVE_FREQ --tex_dim $TEX_DIM --display_env $DISPLAY_NAME --niter $N_ITER --niter_decay $N_ITER_LR_DECAY --dataroot $DATASETS_DIR/$TARGET_ACTOR --model $MODEL --lambda_L1 100 --dataset_mode $DATASET_MODE --gpu_ids $GPUID --lr $LR --batch_size $BATCH_SIZE


#!/bin/bash
set -ex
# bash texture_optimization.sh


DATASETS_DIR=./datasets/
DATASET_MODE=uv
GPUID=0
TARGET_ACTOR=room1


# texture
#TEX_DIM=1024
TEX_DIM=2048

# models
MODEL=RGBTextures

# optimizer parameters
LR=0.01
N_ITER=20
N_ITER_LR_DECAY=40
BATCH_SIZE=1 # has to be 1!!

# save frequency
SAVE_FREQ=10


# loss weights
LAMBDA_L1=10.0
LAMBDA_L1_DIFF=20.0
LAMBDA_VGG=10.0

# regularizer
LAMBDA_REG_TEX=0.1


################################################################################
###############################    TRAINING     ################################
################################################################################

DATE_WITH_TIME=`date "+%Y%m%d-%H%M%S"`
NAME=$MODEL/$TARGET_ACTOR-TEX$TEX_DIM-$LOSS-$DATE_WITH_TIME
DISPLAY_NAME=${MODEL}/${TARGET_ACTOR}-TEX$TEX_DIM

# output directory
RESULT_DIR=./results/$TARGET_ACTOR/TEX$TEX_DIM/
mkdir -p $RESULT_DIR

# training
python train_acc.py --results_dir $RESULT_DIR --name $NAME --save_epoch_freq $SAVE_FREQ --tex_dim $TEX_DIM --lambda_L1 $LAMBDA_L1 --lambda_L1_Diff $LAMBDA_L1_DIFF --lambda_Reg_Tex $LAMBDA_REG_TEX --lambda_VGG $LAMBDA_VGG --display_env $DISPLAY_NAME --niter $N_ITER --niter_decay $N_ITER_LR_DECAY --dataroot $DATASETS_DIR/$TARGET_ACTOR --model $MODEL --dataset_mode $DATASET_MODE --gpu_ids $GPUID --lr $LR --batch_size $BATCH_SIZE


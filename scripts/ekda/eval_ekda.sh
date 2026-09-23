#!/bin/bash

# custom config
DATA= # your directory

DATASET=$1
CFG=$2  # config file
TRAINER=$3
BACKBONE=$4 # backbone name
NTOK=$5
DOMAINS=$6
GPU=$7
KD=$8

LOCATION=middle
TDEEP=False
VDEEP=True

DIR=output/ekda/${TRAINER}/${DATASET}/${CFG}/${BACKBONE//\//}/deep_${LOCATION}/kd${KD}_${DOMAINS}_ntok${NTOK}

python train.py \
    --gpu ${GPU} \
    --backbone ${BACKBONE} \
    --domains ${DOMAINS} \
    --root ${DATA} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir ${DIR} \
    --eval-only \
    TRAINER.EKDA.NUM_TOKENS ${NTOK} \
    TRAINER.EKDA.N_CTX ${NTOK} \
    TRAINER.EKDA.T_DEEP ${TDEEP} \
    TRAINER.EKDA.V_DEEP ${VDEEP} \
    TRAINER.EKDA.LOCATION ${LOCATION} \


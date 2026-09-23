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

TP=True

TDEEP=False
VP=True
VDEEP=True

DIR=output/ekda/${TRAINER}/${DATASET}/${CFG}/${BACKBONE//\//}/deep_${LOCATION}/kd${KD}_${DOMAINS}_ntok${NTOK}

if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}, so skip this job"
else
    echo "Run this job and save the output to ${DIR}"

    python train.py \
        --gpu ${GPU} \
        --kd ${KD} \
        --backbone ${BACKBONE} \
        --domains ${DOMAINS} \
        --root ${DATA} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        TRAINER.EKDA.N_CTX ${NTOK} \
        TRAINER.EKDA.LOCATION ${LOCATION}  \
        TRAINER.EKDA.TP ${TP}\
        TRAINER.EKDA.T_DEEP ${TDEEP} \
        TRAINER.EKDA.VP ${VP} \
        TRAINER.EKDA.V_DEEP ${VDEEP}
fi

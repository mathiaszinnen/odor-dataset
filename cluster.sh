#!/bin/bash -l
#SBATCH --time=14:00:00
#SBATCH --gres=gpu:a100:8
#SBATCH --job-name=person_detection
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

set -e # exit on error to prevent crazy stuff form happening unnoticed

module load python 
module load cuda

echo "execution started"

source activate mmdetection-odor

mkdir ${TMPDIR}/$SLURM_JOB_ID
cd ${TMPDIR}/$SLURM_JOB_ID

cp -r $HOME/odor/odor-dataset .

cd sniffyart-experiments

echo "start copying data"

mkdir -p ./data/odor
tar xf /home/janus/iwi5-datasets/odor/odor3.tar -C ./data/odor

GPUS=0
CONFIG=mmdetection/configs/odor-baselines/fcos_rn50.py
WORK_DIR=$WORK/odor/workdirs/fcos_rn50

mkdir $WORK_DIR

cat "$0" > $WORK_DIR/slurm.sh

echo "train with " ${CONFIG}

srun tools/dist_train.sh ${CONFIG} ${GPUS} --work-dir ${WORK_DIR} 
#python train.py ${CONFIG} --work-dir ${WORK_DIR} 


echo "train done"

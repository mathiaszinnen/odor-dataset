#!/bin/bash -l
#SBATCH --time=14:00:00
#SBATCH --gres=gpu:a40:2
#SBATCH --job-name=fcos_notricks
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

# enable http/s access to download imagenet weights
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

set -e # exit on error to prevent crazy stuff form happening unnoticed

module load python 
module load cuda

echo "execution started"

source activate mmdetection-odor

mkdir ${TMPDIR}/$SLURM_JOB_ID
cd ${TMPDIR}/$SLURM_JOB_ID

cp -r $HOME/odor/odor-dataset .

echo "code repository copied"

cd odor-dataset/FCOS

#echo "start downloading data"

#python prepare_dataset.py



GPUS=2
CONFIG=configs/odor-baselines/fcos_rn50.py
RUN_NUMBER=$1
WORK_DIR=$WORK/odor/workdirs/fcos_notricks/

mkdir $WORK_DIR

cat "$0" > $WORK_DIR/slurm.sh

echo "train with " ${CONFIG}

srun tools/dist_train.sh ${CONFIG} ${GPUS} --work-dir ${WORK_DIR} 
#python train.py ${CONFIG} --work-dir ${WORK_DIR} 


echo "train done"

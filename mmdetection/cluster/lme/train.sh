#!/bin/bash -l

#SBATCH --job-name=FRCNN-swinl
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12000
#SBATCH --gres=gpu:2
#SBATCH -o /home/%u/logs/mmdetection-%x-%j-on-%N.out
#SBATCH -e /home/%u/logs/mmdetection-%x-%j-on-%N.err
#SBATCH --mail-type=ALL
#Timelimit format: "hours:minutes:seconds" -- max is 24h
#SBATCH --time=23:59:59

TMPDIR=/scratch/zinnen/$SLURM_JOB_ID

mkdir -p ${TMPDIR}
cd ${TMPDIR}

# when possible transfer data from /cluster to /scratch, /scratch are located in SSD and data transfer would be faster.
# Please exchange the --your_name-- with your name

cp -r /net/cluster/zinnen/detectors/mmdetection-ODOR .

cd mmdetection-ODOR
mkdir -p data/ODOR-v3

time tar xf /net/cluster/shared_dataset/ODOR/odor3.tar -C ./data/ODOR-v3

source /net/cluster/zinnen/miniconda/etc/profile.d/conda.sh
conda activate openmmlab

#python tools/train.py $1 --work-dir /net/cluster/zinnen/mmdetection-workdirs/$SLURM_JOB_ID --cfg-options epochs=1
./tools/dist_train.sh $1 2 --work-dir /net/cluster/zinnen/mmdetection-workdirs/$SLURM_JOB_ID --cfg-options epochs=1

echo "done"

#rm -rf $TMPDIR

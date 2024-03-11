#!/usr/bin/env bash
set -x

ROOT=../../
export PYTHONPATH=$ROOT:$PYTHONPATH

if [[ ! -d "logs" ]]; then
  mkdir logs
fi

################
gpus=${1-2}
job_name=${2-Hulk_vit-B}
################
partition=${3-Train_partition}
CONFIG=${4-${job_name}.yaml}

g=$((${gpus}<8?${gpus}:8))
echo 'start job:' ${job_name} ' config:' ${CONFIG}

AutoResume=checkpoints/${job_name}/ckpt_task_iter_newest.pth.tar

now=$(date +"%Y%m%d_%H%M%S")
LOG_FILE=logs/${job_name}_${now}.log
echo 'log file: ' ${LOG_FILE}

srun -n${gpus} -p ${partition} --gres=gpu:${g} --ntasks-per-node=${g} --mpi=pmi2 --quotatype=auto  \
    --job-name=${job_name} --cpus-per-task=7  \
    python -W ignore -u ${ROOT}/encoder_decoder.py \
        --expname ${job_name} \
        --config ${CONFIG} \
        --auto-resume=checkpoints/${job_name}/ckpt_task_iter_newest.pth.tar \
        2>&1 | tee ${LOG_FILE}

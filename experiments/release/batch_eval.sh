#!/usr/bin/env bash
#set -x

# source /mnt/lustre/share/spring/r0.3.0
ROOT=../../
export PYTHONPATH=$ROOT:$PYTHONPATH
export JAVA_HOME=/mnt/petrelfs/share/jdk1.8.0_51
export PATH=$JAVA_HOME/bin:$PATH
export CLASSPATH=.:$JAVA_HOME/lib/dt.jar:$JAVA_HOME/lib/tools.jar

if [[ ! -d "logs" ]]; then
  mkdir logs
fi

################
gpus=${1-1}

job_name=${2-debug}
################### ||| additional params usually not used
################### vvv
iter=${3-newest}
PRETRAIN_JOB_NAME=${4-${job_name}}
CONFIG=${5-${job_name}.yaml}
################
g=$((${gpus}<8?${gpus}:8))

#### test list
declare -A test_info_list
test_info_list[2dntu60]=0
test_info_list[smpl]=2
test_info_list[smpl_3dpw]=2
test_info_list[peddet_inter_lpe]=4
test_info_list[rapv2_in_multi_thr0.3]=5
test_info_list[pa100k_in_multi_thr0.3]=5
test_info_list[image_caption]=7
test_info_list[pose_cocoval]=8
test_info_list[pose_aic]=9
test_info_list[par_cihp_flip]=18
test_info_list[par_hm36_flip]=19

for TASK in "${!test_info_list[@]}"; do
  full_job_name=${job_name}_test_${TASK}
  now=$(date +"%Y%m%d_%H%M%S")
  GINFO_INDEX=${test_info_list[${TASK}]}
  LOG_FILE=/mnt/path...to.../Hulk/experiments/release/logs/${full_job_name}_test_${iter}_${TASK}_${now}.out
  echo "=======>${TASK} log file: ${LOG_FILE}"
  TEST_CONFIG=vd_${TASK}_test.yaml
  TEST_MODEL=checkpoints/${PRETRAIN_JOB_NAME}/ckpt_task${GINFO_INDEX}_iter_${iter}.pth.tar
  echo 'start job:' ${full_job_name} ' config:' ${CONFIG} ' test_config:' ${TEST_CONFIG}


  srun -n$1 -p Test  --gres=gpu:${g} --ntasks-per-node=${g}  --quotatype=auto  \
      --job-name=${full_job_name} --cpus-per-task=5  -x SH-IDC1-10-140-24-[13,45,14,78]  \
  python -W ignore -u ${ROOT}/test_mae.py \
      --expname ${full_job_name} \
      --config ${CONFIG} \
      --test_config ${TEST_CONFIG} \
      --spec_ginfo_index ${GINFO_INDEX} \
      --load-path=${TEST_MODEL} \
      2>&1 | tee ${LOG_FILE} 

  sleep 15s
done



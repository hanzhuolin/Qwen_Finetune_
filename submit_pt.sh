#!/bin/bash

#SBATCH --job-name=Qwen_DeepSpeed_Train
#SBATCH --output=/iridisfs/scratch/zh1c23/Qwen/slurm_logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=59:59:00
#SBATCH --partition=a100

# --- ���ؼ��޸� 1�� ---
# ��SLURM�ڳ�ʱǰ60�룬����һ������� SIGUSR1 �źţ�������Ĭ�ϵ� SIGTERM��
#SBATCH --signal=B:USR1@60

# ----- Email notification settings -----
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zh1c23@soton.ac.uk

# --- �����źŴ����� ---

RESUBMIT_FLAG="/iridisfs/scratch/zh1c23/Qwen/slurm_logs/resubmit_${SLURM_JOB_ID}.flag"

# �����յ�����Լ���� SIGUSR1 �ź�ʱ��ִ�д˺���
handle_timeout_and_resubmit() {
  echo "--- Caught SIGUSR1: SLURM time limit is approaching! ---"
  echo "--- Creating resubmit flag and submitting the next job. ---"
  touch "${RESUBMIT_FLAG}"
  sbatch submit_pt.sh
  sleep 5
}

# --- ���ؼ��޸� 2�� ---
# �� trap ֻ��������� SIGUSR1 �źš�
# ���������ֶ� scancel ���͵�Ĭ�� SIGTERM �źžͲ��ᱻ�������ˡ�
trap 'handle_timeout_and_resubmit' USR1

# --- ����ִ�����壨�ⲿ������Ķ��� ---
echo "======================================================"
echo "Model training task started on: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Trap for SIGUSR1 is set. Script PID: $$"
echo "======================================================"

# ��ʼ������
source /iridisfs/scratch/zh1c23/anaconda3/etc/profile.d/conda.sh
conda activate Qwen
module load cuda/11.8.0
export LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8
export LD_LIBRARY_PATH=/iridisfs/scratch/zh1c23/anaconda3/envs/Qwen/lib:$LD_LIBRARY_PATH
cd /iridisfs/scratch/zh1c23/Qwen
echo "Current working directory: $(pwd)"

CHECKPOINT_PATH="/iridisfs/scratch/zh1c23/Qwen/results/pt_checkpoints"

if [ -f "${CHECKPOINT_PATH}/training_completed.flag" ]; then
    echo "--- Training has already been completed. Exiting now. ---"
    exit 0
fi

LOG_FILE="logs/train_model_${SLURM_JOB_ID}.log"
echo "--- Starting training in the background. Log will be written to ${LOG_FILE} ---"

accelerate launch --config_file lora_script.yaml > "${LOG_FILE}" 2>&1 &

pid=$!
echo "Training process started with PID: $pid"

wait $pid

EXIT_CODE=$?
echo "--- Training process (PID: $pid) finished with Exit Code: ${EXIT_CODE} ---"

# --- ���������������ж� ---

if [ -f "${RESUBMIT_FLAG}" ]; then
    echo "--- Resubmit flag found. This job was terminated due to a timeout. ---"
    echo "--- The next job has already been submitted by the trap handler. ---"
    rm "${RESUBMIT_FLAG}"
else
    # ֻ���ڽű��Լ������������ǳɹ�����ʧ�ܣ�ʱ�Ż��ߵ�����
    # ��Ϊ�ֶ�scancel(SIGTERM)���ᱻtrap����Ҳ���ߵ�����
    if [ ${EXIT_CODE} -eq 0 ]; then
        echo "--- Training completed successfully (Exit Code 0). Not resubmitting. ---"
    else
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "!!! TRAINING SCRIPT FAILED or was MANUALLY CANCELLED (Exit Code: ${EXIT_CODE})"
        echo "!!! This was an application error or a manual scancel, NOT a timeout."
        echo "!!! THE JOB CHAIN WILL BE STOPPED."
        echo "!!! Check log for details: ${LOG_FILE}"
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    fi
fi

echo "======================================================"
echo "Task for Job ID $SLURM_JOB_ID completed on: $(date)"
echo "======================================================"
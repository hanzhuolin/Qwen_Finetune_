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

# --- 【关键修改 1】 ---
# 让SLURM在超时前60秒，发送一个特殊的 SIGUSR1 信号，而不是默认的 SIGTERM。
#SBATCH --signal=B:USR1@60

# ----- Email notification settings -----
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zh1c23@soton.ac.uk

# --- 定义信号处理函数 ---

RESUBMIT_FLAG="/iridisfs/scratch/zh1c23/Qwen/slurm_logs/resubmit_${SLURM_JOB_ID}.flag"

# 当接收到我们约定的 SIGUSR1 信号时，执行此函数
handle_timeout_and_resubmit() {
  echo "--- Caught SIGUSR1: SLURM time limit is approaching! ---"
  echo "--- Creating resubmit flag and submitting the next job. ---"
  touch "${RESUBMIT_FLAG}"
  sbatch submit_pt.sh
  sleep 5
}

# --- 【关键修改 2】 ---
# 让 trap 只捕获特殊的 SIGUSR1 信号。
# 这样，你手动 scancel 发送的默认 SIGTERM 信号就不会被它捕获了。
trap 'handle_timeout_and_resubmit' USR1

# --- 任务执行主体（这部分无需改动） ---
echo "======================================================"
echo "Model training task started on: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Trap for SIGUSR1 is set. Script PID: $$"
echo "======================================================"

# 初始化环境
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

# --- 任务结束后的最终判断 ---

if [ -f "${RESUBMIT_FLAG}" ]; then
    echo "--- Resubmit flag found. This job was terminated due to a timeout. ---"
    echo "--- The next job has already been submitted by the trap handler. ---"
    rm "${RESUBMIT_FLAG}"
else
    # 只有在脚本自己结束（无论是成功还是失败）时才会走到这里
    # 因为手动scancel(SIGTERM)不会被trap捕获，也会走到这里
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
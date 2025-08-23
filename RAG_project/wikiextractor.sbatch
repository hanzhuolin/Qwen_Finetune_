#!/bin/bash
#=========================================================================================
# SLURM Job Submission Script for WikiExtractor
#=========================================================================================
# -- Job Details --
#SBATCH --job-name=wikiextractor         # 任务名称，方便识别
#SBATCH --output=/iridisfs/scratch/zh1c23/Qwen/Rag/logs/wikiextractor-%j.out  # 标准输出日志
#SBATCH --error=/iridisfs/scratch/zh1c23/Qwen/Rag/logs/wikiextractor-%j.err   # 标准错误日志

# -- Resource Allocation --
#SBATCH --time=12:00:00                  # 运行时间 (4小时)
#SBATCH --partition=amd                 # 使用 amd CPU 分区
#SBATCH --nodes=1                       # 使用单个节点
#SBATCH --ntasks-per-node=1             # 每个节点运行一个任务
#SBATCH --cpus-per-task=16              # 为每个任务分配16个CPU核心
#SBATCH --mem=128G                       # 分配32GB内存

# -- Optional: Email Notifications --
#SBATCH --mail-type=ALL                 # 在任务开始、结束或失败时发送邮件
#SBATCH --mail-user=your_email@example.com # 【可选】替换为您的邮箱地址

#=========================================================================================
# Job Execution
#=========================================================================================

# --- 1. Preamble ---
echo "======================================================"
echo "Job started on: $(date)"
echo "Job running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "======================================================"

# --- 2. Environment Setup ---
source /iridisfs/scratch/zh1c23/anaconda3/etc/profile.d/conda.sh

# 激活您的Conda环境
echo "Activating Conda environment: rag-wiki"
conda activate rag-wiki

# --- 3. Create directories and check input ---
# 创建必要的输出目录
mkdir -p /iridisfs/scratch/zh1c23/Qwen/Rag/data/extracted
mkdir -p /iridisfs/scratch/zh1c23/Qwen/Rag/logs

# 定义输入输出路径
INPUT_FILE="/iridisfs/scratch/zh1c23/Qwen/Rag/data/enwiki-latest-pages-articles.xml.bz2"
OUTPUT_DIR="/iridisfs/scratch/zh1c23/Qwen/Rag/data/extracted"

echo "Input file: $INPUT_FILE"
echo "Output directory: $OUTPUT_DIR"

# 检查输入文件是否存在
if [ ! -f "$INPUT_FILE" ]; then
    echo "ERROR: Input file not found: $INPUT_FILE"
    exit 1
fi

echo "Input file size: $(du -h $INPUT_FILE)"
echo "======================================================"

# --- 4. Install dependencies ---
echo "Checking wikiextractor installation..."
pip show wikiextractor &>/dev/null || pip install wikiextractor

# 显示wikiextractor版本和帮助信息
echo "WikiExtractor version and available options:"
python -m wikiextractor.WikiExtractor --help | head -20

echo "======================================================"
echo "Starting WikiExtractor processing..."
echo "======================================================"

# --- 5. Run WikiExtractor ---
# 使用简化的参数设置
python -m wikiextractor.WikiExtractor \
    --json \
    --processes 16 \
    -b 500M \
    -o "$OUTPUT_DIR" \
    "$INPUT_FILE"

# --- 6. Check results ---
EXTRACTION_EXIT_CODE=$?

if [ $EXTRACTION_EXIT_CODE -eq 0 ]; then
    echo "======================================================"
    echo "WikiExtractor completed successfully!"
    echo "======================================================"
    
    # 显示提取结果
    echo "Output directory contents:"
    ls -la "$OUTPUT_DIR"
    
    echo ""
    echo "Number of wiki files generated:"
    find "$OUTPUT_DIR" -name "wiki_*" -type f | wc -l
    
    echo ""
    echo "Total size of extracted data:"
    du -sh "$OUTPUT_DIR"
    
    echo ""
    echo "Sample of generated files:"
    find "$OUTPUT_DIR" -name "wiki_*" -type f | head -10
    
    # 显示内容样例
    echo ""
    echo "======================================================"
    echo "Sample content from first file:"
    echo "======================================================"
    FIRST_FILE=$(find "$OUTPUT_DIR" -name "wiki_*" -type f | head -1)
    if [ -n "$FIRST_FILE" ]; then
        echo "File: $FIRST_FILE"
        if [[ "$FIRST_FILE" == *.gz ]]; then
            zcat "$FIRST_FILE" | head -3
        else
            head -3 "$FIRST_FILE"
        fi
    fi
    
else
    echo "ERROR: WikiExtractor failed with exit code $EXTRACTION_EXIT_CODE"
    exit $EXTRACTION_EXIT_CODE
fi

# --- 7. Epilogue ---
echo "======================================================"
echo "Job finished with exit code $EXTRACTION_EXIT_CODE on: $(date)"
echo "======================================================"
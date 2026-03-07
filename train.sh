#!/bin/bash

# Training Script - Improved based on successful notebook
# Sử dụng config và workflow từ notebook có accuracy 90%+

set -e

# Auto detect base dir
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║      PaddleOCR Vietnamese - Improved Training              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "📁 Base dir: $BASE_DIR"

# Step 1: Verify data
echo -e "\n${BLUE}[1/5]${NC} ${GREEN}Verifying data...${NC}"
if [ ! -f "$BASE_DIR/data/train_list.txt" ]; then
    echo -e "${YELLOW}❌ data/train_list.txt not found!${NC}"
    echo "Please run: python prepare_data.py first"
    exit 1
fi

train_count=$(wc -l < "$BASE_DIR/data/train_list.txt")
val_count=$(wc -l < "$BASE_DIR/data/val_list.txt")

echo "✓ Train samples: ${train_count}"
echo "✓ Val samples: ${val_count}"

# Step 2: Download pretrained if not exists
echo -e "\n${BLUE}[2/5]${NC} ${GREEN}Checking pretrained model...${NC}"
mkdir -p "$BASE_DIR/pretrain_models"

if [ ! -f "$BASE_DIR/pretrain_models/PP-OCRv5_mobile_rec_pretrained.pdparams" ]; then
    echo "Downloading pretrained model..."
    cd "$BASE_DIR/pretrain_models"
    wget -q --show-progress https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/latin_PP-OCRv5_mobile_rec_pretrained.pdparams \
        -O PP-OCRv5_mobile_rec_pretrained.pdparams
    cd "$BASE_DIR"
    echo "✓ Downloaded"
else
    echo "✓ Pretrained model exists"
fi

# Step 3: Fix config paths to match current directory
echo -e "\n${BLUE}[3/5]${NC} ${GREEN}Fixing config paths...${NC}"
python "$BASE_DIR/prepare_data.py" \
    --input_dir /dev/null \
    --output_dir "$BASE_DIR/data" \
    --fix_config "$BASE_DIR/config.yml" \
    --base_dir "$BASE_DIR"
echo "✓ Config paths fixed"

# Step 4: Train
echo -e "\n${BLUE}[4/5]${NC} ${GREEN}Starting training...${NC}"
echo -e "${YELLOW}Config: $BASE_DIR/config.yml${NC}"
echo -e "${YELLOW}Epochs: 100${NC}"
echo -e "${YELLOW}Batch size: 128${NC}"
echo ""

LOG_FILE="logs/train_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$BASE_DIR/logs"

cd "$BASE_DIR/PaddleOCR"

python -m paddle.distributed.launch \
    --gpus '0,1' \
    tools/train.py \
    -c "$BASE_DIR/config.yml" \
    2>&1 | tee "$BASE_DIR/$LOG_FILE"

cd "$BASE_DIR"

# Step 5: Show results
echo -e "\n${BLUE}[5/5]${NC} ${GREEN}Training completed!${NC}"
echo ""
echo "📊 Last 10 accuracy scores:"
tail -50 "$BASE_DIR/$LOG_FILE" | grep -i "acc:" | tail -10

echo ""
echo "✅ Model saved to: $BASE_DIR/output/vi_ppocr_v5/"
echo "📝 Log file: $BASE_DIR/$LOG_FILE"

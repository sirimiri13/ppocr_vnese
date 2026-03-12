#!/bin/bash

# Training Script - Improved based on successful notebook
# Sử dụng config và workflow từ notebook có accuracy 90%+

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║      PaddleOCR Vietnamese - Improved Training              ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# Step 1: Verify data
echo -e "\n${BLUE}[1/5]${NC} ${GREEN}Verifying data...${NC}"
if [ ! -f "data/train_list.txt" ]; then
    echo -e "${YELLOW}❌ data/train_list.txt not found!${NC}"
    echo "Please run: python fix_prepare_data.py first"
    exit 1
fi

train_count=$(wc -l < data/train_list.txt)
val_count=$(wc -l < data/val_list.txt)

echo "✓ Train samples: ${train_count}"
echo "✓ Val samples: ${val_count}"

# Step 2: Download pretrained if not exists
echo -e "\n${BLUE}[2/5]${NC} ${GREEN}Checking pretrained model...${NC}"
mkdir -p pretrain_models

if [ ! -f "pretrain_models/PP-OCRv5_mobile_rec_pretrained.pdparams" ]; then
    echo "Downloading pretrained model..."
    cd pretrain_models
    wget -q --show-progress https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/latin_PP-OCRv5_mobile_rec_pretrained.pdparams \
        -O PP-OCRv5_mobile_rec_pretrained.pdparams
    cd ..
    echo "✓ Downloaded"
else
    echo "✓ Pretrained model exists"
fi

# Step 3: Fix config paths
echo -e "\n${BLUE}[3/5]${NC} ${GREEN}Fixing config paths...${NC}"
if [ -f "config.yml" ]; then
    python -c "
import yaml, os
with open('config.yml', 'r') as f:
    cfg = yaml.safe_load(f)

base = '/kaggle/working/paddleocr-v5-vietnamese'
cfg['Global']['character_dict_path'] = os.path.join(base, 'dict/vi_dict.txt')
cfg['Global']['pretrained_model'] = os.path.join(base, 'pretrain_models/latin_PP-OCRv5_mobile_rec_pretrained.pdparams')
cfg['Global']['save_model_dir'] = os.path.join(base, 'output/vi_ppocr_v5')
cfg['Train']['dataset']['data_dir'] = os.path.join(base, 'data/')
cfg['Train']['dataset']['label_file_list'] = [os.path.join(base, 'data/train_list.txt')]
cfg['Eval']['dataset']['data_dir'] = os.path.join(base, 'data/')
cfg['Eval']['dataset']['label_file_list'] = [os.path.join(base, 'data/val_list.txt')]

with open('config.yml', 'w') as f:
    yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)
print('✓ Config paths fixed')
    "
fi
echo "✓ Config ready"

# Step 4: Train
echo -e "\n${BLUE}[4/5]${NC} ${GREEN}Starting training...${NC}"
echo -e "${YELLOW}Config: config.yml${NC}"
echo -e "${YELLOW}Epochs: 100${NC}"
echo -e "${YELLOW}Batch size: 128${NC}"
echo ""

LOG_FILE="logs/train_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

cd PaddleOCR

python -m paddle.distributed.launch \
    --gpus '0,1' \
    tools/train.py \
    -c /kaggle/working/paddleocr-v5-vietnamese/config.yml \
    2>&1 | tee /kaggle/working/paddleocr-v5-vietnamese/$LOG_FILE

cd ..

# Step 5: Show results
echo -e "\n${BLUE}[5/5]${NC} ${GREEN}Training completed!${NC}"
echo ""
echo "📊 Last 10 accuracy scores:"
tail -50 $LOG_FILE | grep -i "acc:" | tail -10

echo ""
echo "✅ Model saved to: output/vi_ppocr_v5/"
echo "📝 Log file: $LOG_FILE"

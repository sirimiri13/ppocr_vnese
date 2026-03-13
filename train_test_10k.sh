#!/bin/bash

# Training Script - Test với 10k samples
# Optimized settings: width=640, max_len=96, batch=8

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║      PaddleOCR Vietnamese - Test 10k (Optimized)           ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# Step 1: Verify data
echo -e "\n${BLUE}[1/5]${NC} ${GREEN}Verifying data...${NC}"
if [ ! -f "data/train_list.txt" ]; then
    echo -e "${YELLOW}❌ data/train_list.txt not found!${NC}"
    echo "Please run: python prepare_data.py first"
    exit 1
fi

train_count=$(wc -l < data/train_list.txt)
val_count=$(wc -l < data/val_list.txt)

echo "✓ Train samples: ${train_count}"
echo "✓ Val samples: ${val_count}"

# Limit to 10k for test
if [ $train_count -gt 10000 ]; then
    echo -e "${YELLOW}⚠️  Limiting to 10k samples for test...${NC}"
    head -10000 data/train_list.txt > data/train_list_10k.txt
    mv data/train_list_10k.txt data/train_list.txt
    echo "✓ Limited to 10k samples"
fi

# Step 2: Check pretrained
echo -e "\n${BLUE}[2/5]${NC} ${GREEN}Checking pretrained model...${NC}"
if [ ! -f "/kaggle/working/ppocr_vnese/pretrain_models/PP-OCRv5_mobile_rec_pretrained.pdparams" ]; then
    echo -e "${YELLOW}❌ Pretrained model not found!${NC}"
    echo "Please run: bash setup_kaggle.sh first"
    exit 1
fi
echo "✓ Pretrained model exists"

# Step 3: Fix config paths (if needed)
echo -e "\n${BLUE}[3/5]${NC} ${GREEN}Fixing config paths...${NC}"
if [ -f "config_test_10k.yml" ]; then
    python -c "
import yaml, os
with open('config_test_10k.yml', 'r') as f:
    cfg = yaml.safe_load(f)

base = '/kaggle/working/ppocr_vnese'

cfg['Global']['character_dict_path'] = os.path.join(base, 'dict/vi_dict.txt')
cfg['Global']['pretrained_model'] = os.path.join(base, 'pretrain_models/PP-OCRv5_mobile_rec_pretrained.pdparams')
cfg['Global']['save_model_dir'] = os.path.join(base, 'output/test_10k')
cfg['Train']['dataset']['data_dir'] = os.path.join(base, 'data/')
cfg['Train']['dataset']['label_file_list'] = [os.path.join(base, 'data/train_list.txt')]
cfg['Eval']['dataset']['data_dir'] = os.path.join(base, 'data/')
cfg['Eval']['dataset']['label_file_list'] = [os.path.join(base, 'data/val_list.txt')]

with open('config_test_10k.yml', 'w') as f:
    yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)
print('✓ Config paths fixed')
    "
fi
echo "✓ Config ready"

# Step 4: Train
echo -e "\n${BLUE}[4/5]${NC} ${GREEN}Starting test training...${NC}"
echo -e "${YELLOW}Settings:${NC}"
echo "  - Config: config_test_10k.yml"
echo "  - Samples: 10k"
echo "  - Epochs: 30"
echo "  - Image width: 640"
echo "  - Max length: 96"
echo "  - Batch size: 8"
echo ""

LOG_FILE="logs/test_10k_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

cd PaddleOCR

python -m paddle.distributed.launch \
    --gpus '0,1' \
    tools/train.py \
    -c /kaggle/working/ppocr_vnese/config_test_10k.yml \
    2>&1 | tee /kaggle/working/ppocr_vnese/$LOG_FILE

cd ..

# Step 5: Show results
echo -e "\n${BLUE}[5/5]${NC} ${GREEN}Training completed!${NC}"
echo ""
echo "📊 Last 10 accuracy scores:"
tail -50 $LOG_FILE | grep -i "acc:" | tail -10

echo ""
echo "✅ Model saved to: output/test_10k/"
echo "📝 Log file: $LOG_FILE"
echo ""
echo "🎯 Expected results:"
echo "  - Epoch 10: ~60-70% accuracy"
echo "  - Epoch 20: ~75-80% accuracy"
echo "  - Epoch 30: ~80-85% accuracy"
echo ""
echo "Next steps:"
echo "  1. Check results: tail -50 $LOG_FILE | grep acc"
echo "  2. Test model: See README.md"
echo "  3. Train full: bash train.sh (for >90% accuracy)"

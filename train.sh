#!/bin/bash

# Training Script - Hỗ trợ resume checkpoint trên Kaggle
# Kaggle timeout 12h → save checkpoint → chạy session mới → resume tiếp

set -e

# Auto detect base dir
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║      PaddleOCR Vietnamese - Training (Resume Support)      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "📁 Base dir: $BASE_DIR"

# ============================================================
# Step 1: Verify data
# ============================================================
echo -e "\n${BLUE}[1/6]${NC} ${GREEN}Verifying data...${NC}"
if [ ! -f "$BASE_DIR/data/train_list.txt" ]; then
    echo -e "${RED}❌ data/train_list.txt not found!${NC}"
    echo "Please run: python prepare_data.py first"
    exit 1
fi

train_count=$(wc -l < "$BASE_DIR/data/train_list.txt")
val_count=$(wc -l < "$BASE_DIR/data/val_list.txt")
echo "✓ Train samples: ${train_count}"
echo "✓ Val samples: ${val_count}"

# ============================================================
# Step 2: Download pretrained if not exists
# ============================================================
echo -e "\n${BLUE}[2/6]${NC} ${GREEN}Checking pretrained model...${NC}"
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

# ============================================================
# Step 3: Restore checkpoint từ session trước (nếu có)
# ============================================================
echo -e "\n${BLUE}[3/6]${NC} ${GREEN}Checking for previous checkpoints...${NC}"
CHECKPOINT_DIR="$BASE_DIR/output/vi_ppocr_v5"
mkdir -p "$CHECKPOINT_DIR"

# Tìm checkpoint zip từ Kaggle input (output của session trước)
# User cần add output của notebook trước làm input
CHECKPOINT_ZIP=""
for candidate in \
    "/kaggle/input/ppocr-checkpoint/checkpoint_vi_ppocr_v5.zip" \
    "/kaggle/input/ppocr-vnese-checkpoint/checkpoint_vi_ppocr_v5.zip" \
    "/kaggle/input/checkpoint/checkpoint_vi_ppocr_v5.zip"; do
    if [ -f "$candidate" ]; then
        CHECKPOINT_ZIP="$candidate"
        break
    fi
done

# Cũng tìm từ output notebook (nếu save as dataset)
if [ -z "$CHECKPOINT_ZIP" ]; then
    FOUND=$(find /kaggle/input -name "checkpoint_vi_ppocr_v5.zip" 2>/dev/null | head -1)
    if [ -n "$FOUND" ]; then
        CHECKPOINT_ZIP="$FOUND"
    fi
fi

if [ -n "$CHECKPOINT_ZIP" ] && [ ! -f "${CHECKPOINT_DIR}/latest.pdparams" ]; then
    echo -e "${GREEN}📦 Tìm thấy checkpoint từ session trước: $CHECKPOINT_ZIP${NC}"
    echo "   Đang restore..."
    unzip -o "$CHECKPOINT_ZIP" -d "$CHECKPOINT_DIR/"
    echo "   ✓ Checkpoint restored!"
elif [ -f "${CHECKPOINT_DIR}/latest.pdparams" ]; then
    echo "✓ Checkpoint đã có sẵn trong output dir"
else
    echo "ℹ️  Không có checkpoint cũ → sẽ train từ pretrained model"
fi

# ============================================================
# Step 4: Fix config paths
# ============================================================
echo -e "\n${BLUE}[4/6]${NC} ${GREEN}Fixing config paths...${NC}"
python "$BASE_DIR/prepare_data.py" \
    --input_dir /dev/null \
    --output_dir "$BASE_DIR/data" \
    --fix_config "$BASE_DIR/config.yml" \
    --base_dir "$BASE_DIR"
echo "✓ Config paths fixed"

# ============================================================
# Step 5: Train
# ============================================================
echo -e "\n${BLUE}[5/6]${NC} ${GREEN}Starting training...${NC}"
echo -e "${YELLOW}Config: $BASE_DIR/config.yml${NC}"
echo -e "${YELLOW}Epochs: 200${NC}"
echo -e "${YELLOW}Batch size: 32${NC}"

# Verify critical files
echo ""
echo "🔍 Verifying paths..."
for f in "$BASE_DIR/dict/vi_dict.txt" \
         "$BASE_DIR/data/train_list.txt" \
         "$BASE_DIR/data/val_list.txt" \
         "$BASE_DIR/pretrain_models/PP-OCRv5_mobile_rec_pretrained.pdparams"; do
    if [ -f "$f" ]; then
        echo "  ✓ $(basename $f)"
    else
        echo "  ❌ MISSING: $f"
        exit 1
    fi
done

# Quyết định: resume checkpoint hay train từ pretrained
CHECKPOINT_OPT=""
if [ -f "${CHECKPOINT_DIR}/latest.pdparams" ]; then
    echo ""
    echo -e "${GREEN}🔄 RESUME MODE: Tiếp tục từ checkpoint${NC}"
    echo "   Checkpoint: ${CHECKPOINT_DIR}/latest"
    CHECKPOINT_OPT="-o Global.checkpoints=${CHECKPOINT_DIR}/latest"
else
    echo ""
    echo -e "${YELLOW}🆕 NEW MODE: Train từ pretrained Latin model${NC}"
    CHECKPOINT_OPT="-o Global.pretrained_model=$BASE_DIR/pretrain_models/PP-OCRv5_mobile_rec_pretrained.pdparams"
fi
echo ""

LOG_FILE="logs/train_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$BASE_DIR/logs"

cd "$BASE_DIR/PaddleOCR"

python -m paddle.distributed.launch \
    --gpus '0,1' \
    tools/train.py \
    -c "$BASE_DIR/config.yml" \
    ${CHECKPOINT_OPT} \
    -o Global.character_dict_path="$BASE_DIR/dict/vi_dict.txt" \
    -o Global.save_model_dir="$CHECKPOINT_DIR" \
    -o Global.save_res_path="$BASE_DIR/output/rec/predicts_vi.txt" \
    -o Train.dataset.data_dir="$BASE_DIR/data/" \
    -o Train.dataset.label_file_list="['$BASE_DIR/data/train_list.txt']" \
    -o Eval.dataset.data_dir="$BASE_DIR/data/" \
    -o Eval.dataset.label_file_list="['$BASE_DIR/data/val_list.txt']" \
    -o Train.loader.batch_size_per_card=32 \
    -o Train.sampler.first_bs=32 \
    -o Train.loader.num_workers=4 \
    2>&1 | tee "$BASE_DIR/$LOG_FILE"

cd "$BASE_DIR"

# ============================================================
# Step 6: Save checkpoint để resume session sau
# ============================================================
echo -e "\n${BLUE}[6/6]${NC} ${GREEN}Saving checkpoint...${NC}"

# Zip checkpoint lại → Kaggle giữ /kaggle/working/ output
if [ -f "${CHECKPOINT_DIR}/latest.pdparams" ]; then
    echo "📦 Zipping checkpoint..."
    cd "$CHECKPOINT_DIR"
    zip -q /kaggle/working/checkpoint_vi_ppocr_v5.zip \
        latest.pdparams latest.pdopt latest.states \
        best_accuracy.pdparams best_accuracy.pdopt best_accuracy.states \
        2>/dev/null || true
    cd "$BASE_DIR"
    echo "✓ Checkpoint saved: /kaggle/working/checkpoint_vi_ppocr_v5.zip"
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  📌 ĐỂ RESUME Ở SESSION SAU:                              ║"
    echo "║  1. Save notebook output                                   ║"
    echo "║  2. Tạo notebook mới → Add Data → chọn output notebook cũ  ║"
    echo "║  3. Chạy lại train.sh → tự động resume từ checkpoint       ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
else
    echo "⚠️  Không tìm thấy checkpoint (có thể training bị crash)"
fi

# Show results
echo ""
echo "📊 Last 10 accuracy scores:"
tail -50 "$BASE_DIR/$LOG_FILE" 2>/dev/null | grep -i "acc:" | tail -10 || echo "(no accuracy logs found)"

echo ""
echo "✅ Model saved to: $CHECKPOINT_DIR/"
echo "📝 Log file: $BASE_DIR/$LOG_FILE"

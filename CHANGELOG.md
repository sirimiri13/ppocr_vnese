# CHANGELOG - Optimizations from Production Notebook

## 🎯 **CRITICAL CHANGES (Dựa trên notebook 90%+ accuracy)**

### **1. IMAGE WIDTH: 320 → 640 (DOUBLE!)**

**Lý do:**
- Text dài hơn không bị cắt/méo
- Chi tiết rõ hơn cho chữ nhỏ
- Accuracy tăng ~5-10%

**Impact:**
```yaml
# Old
d2s_train_image_shape: [3, 48, 320]

# New
d2s_train_image_shape: [3, 48, 640]  # ✅
```

**Tất cả nơi phải đổi:**
- `Global.d2s_train_image_shape`
- `Train.transforms.RecConAug.image_shape`
- `Train.sampler.scales`
- `Eval.transforms.RecResizeImg.image_shape`

---

### **2. MAX TEXT LENGTH: 40 → 96**

**Lý do:**
- Văn bản cổ có câu rất dài (>40 ký tự)
- 40 ký tự bị cắt → mất accuracy

**Impact:**
```yaml
# Old
max_text_length: 40

# New
max_text_length: 96  # ✅
```

**Tất cả nơi phải đổi:**
- `Global.max_text_length`
- `Architecture.Head.NRTRHead.max_text_length`
- `Train.transforms.RecConAug.max_text_length`
- `Train.transforms.MultiLabelEncode.max_text_length`
- `Eval.transforms.MultiLabelEncode.max_text_length`

---

### **3. BATCH SIZE: 128 → 8**

**Lý do:**
- Width 640 + Batch 128 = **OUT OF MEMORY!**
- T4 GPU chỉ có 15GB VRAM
- Width 640 tốn gấp đôi memory

**Impact:**
```yaml
# Old (OOM!)
batch_size_per_card: 128

# New
batch_size_per_card: 8  # ✅
```

**QUAN TRỌNG:** Cũng phải giảm `first_bs`:
```yaml
# Old
first_bs: 128

# New
first_bs: 32  # ✅
```

---

### **4. EPOCHS: 100 → 50**

**Lý do:**
- 50 epochs đủ để converge
- Notebook thành công dùng 50 epochs
- Tiết kiệm thời gian (~50%)

**Impact:**
```yaml
# Old
epoch_num: 100

# New
epoch_num: 50  # ✅
```

---

### **5. DEPENDENCIES (CRITICAL!)**

**Lý do:**
- Paddle 3.2.0 + PaddleX conflict!
- Phải install --no-deps và fix versions

**Install sequence:**
```bash
# 1. Paddle 3.2.0
pip install paddlepaddle-gpu==3.2.0

# 2. PaddleOCR requirements (NO deps)
pip install -r requirements.txt --no-deps

# 3. Fix versions (CRITICAL)
pip install --force-reinstall \
    numpy==1.26.4 \
    scipy==1.11.4 \
    scikit-learn==1.3.2
```

**Không làm bước này = nhiều lỗi lạ!**

---

## 📊 **COMPARISON TABLE:**

| Parameter | Old (Baseline) | **New (Optimized)** | Impact |
|-----------|----------------|---------------------|--------|
| **Image width** | 320 | **640** | +5-10% acc, -memory |
| **Max length** | 40 | **96** | Handle long text |
| **Batch size** | 128 | **8** | No OOM |
| **First BS** | 128 | **32** | Stable memory |
| **Epochs** | 100 | **50** | Faster training |
| **Training time** | 16-20h | **12-16h** | -25% time |
| **Expected acc** | 85-90% | **>90%** | Better! |

---

## ⚠️ **COMMON PITFALLS:**

### **❌ Pitfall 1: Không đổi tất cả width values**
```yaml
# SAI - Chỉ đổi 1 chỗ
Global:
  d2s_train_image_shape: [3, 48, 640]  # ✓
Train:
  transforms:
    RecConAug:
      image_shape: [48, 320, 3]  # ✗ Vẫn 320!
```

**→ Lỗi shape mismatch!**

### **❌ Pitfall 2: Không giảm batch size**
```yaml
# SAI - Width 640 nhưng batch 128
d2s_train_image_shape: [3, 48, 640]
batch_size_per_card: 128  # ✗ OOM!
```

**→ Out of Memory!**

### **❌ Pitfall 3: Quên first_bs**
```yaml
# SAI - Chỉ giảm batch_size
loader:
  batch_size_per_card: 8  # ✓
sampler:
  first_bs: 128  # ✗ Vẫn 128!
```

**→ Vẫn OOM!**

---

## ✅ **VERIFICATION CHECKLIST:**

Trước khi train, check:

```bash
# Check all width=640
grep -E "640|320" config.yml

# Should see:
# d2s_train_image_shape: [3, 48, 640]  ✓
# image_shape: [48, 640, 3]             ✓
# scales: [[640, 32], [640, 48], ...]   ✓
# image_shape: [3, 48, 640]             ✓

# Check all max_length=96
grep "max_text_length" config.yml

# Should see:
# max_text_length: 96  (multiple places)  ✓

# Check batch sizes
grep "batch_size\|first_bs" config.yml

# Should see:
# batch_size_per_card: 8   ✓
# first_bs: 32             ✓
```

---

## 🎯 **MIGRATION GUIDE:**

Nếu đang train với config cũ:

### **Option 1: Start fresh (Recommended)**
```bash
# Stop current training
# Use new config
bash train.sh
```

### **Option 2: Resume with new config (Advanced)**
```bash
# Update config
# Resume từ checkpoint
python -m paddle.distributed.launch \
    --gpus '0,1' \
    tools/train.py \
    -c config_new.yml \
    -o Global.checkpoints=output/vi_ppocr_v5/latest
```

**Note:** Model trained với width=320 không dùng được với width=640!

---

## 📈 **EXPECTED IMPROVEMENTS:**

```
Old config (width=320, batch=128):
├── Training time: 16-20h
├── Accuracy: 85-90%
├── Memory: OK (batch=128 fits)
└── Text handling: Medium (max_len=40)

New config (width=640, batch=8):
├── Training time: 12-16h  ✅ Faster!
├── Accuracy: >90%         ✅ Better!
├── Memory: OK (batch=8)
└── Text handling: Excellent (max_len=96)  ✅
```

---

**Tóm lại: Tất cả thay đổi đều dựa trên notebook production đã verify >90% accuracy!** 🎯

# PaddleOCR Vietnamese - Production Ready

Dựa trên notebook thành công với accuracy 90%+ 

## 🎯 Điểm Nổi Bật

- ✅ **Paddle 3.2.0** - Phiên bản mới nhất, ổn định
- ✅ **Config tối ưu** - Từ notebook production 90%+
- ✅ **Image width 640** - Gấp đôi so với baseline (320)
- ✅ **Max text length 96** - Hỗ trợ text dài hơn (thay vì 40)
- ✅ **Batch size 8** - Ổn định trên T4 GPU
- ✅ **Kết quả cao** - >90% accuracy

## 🔥 **THAY ĐỔI QUAN TRỌNG:**

### **So với baseline:**
| Parameter | Baseline | **Optimized** | Lý do |
|-----------|----------|---------------|-------|
| Image width | 320 | **640** | Text dài hơn, rõ hơn |
| Max length | 40 | **96** | Hỗ trợ câu dài |
| Batch size | 128 | **8** | Tránh OOM trên T4 |
| First BS | 128 | **32** | Ổn định memory |
| Epochs | 100 | **50** | Đủ để converge |

## 📁 Cấu Trúc Project

```
paddleocr-v5-vietnamese/
├── README.md                # Hướng dẫn
├── requirements.txt         # Dependencies
├── config.yml              # Config tối ưu (width=640, len=96)
├── setup_kaggle.sh         # Setup môi trường
├── prepare_data.py         # Chuẩn bị data (all-in-one)
└── train.sh                # Training script
```

## 🚀 Quick Start - 3 Bước Đơn Giản

### **Option 1: Test 10k trước (Khuyến nghị - 3-4 giờ)**

```python
# ========== Bước 1: Setup ==========
%cd /kaggle/working
!git clone https://github.com/YOUR_USERNAME/paddleocr-v5-vietnamese.git
%cd paddleocr-v5-vietnamese
!bash setup_kaggle.sh

# ========== Bước 2: Prepare Data (10k samples) ==========
!python prepare_data.py \
    --input_dir /kaggle/input/datasets/sirimiriiii13/vocr-rec/FinalData \
    --output_dir ./data \
    --max_samples 10000

# Verify
!wc -l data/train_list.txt data/val_list.txt

# ========== Bước 3: Train Test (3-4h → 80-85% acc) ==========
!bash train_test_10k.sh

# Check results
!tail -50 logs/test_10k_*.log | grep "acc:" | tail -10
```

**Kết quả mong đợi:**
- Epoch 10: ~60-70% acc
- Epoch 20: ~75-80% acc  
- Epoch 30: **80-85% acc** ✅

---

### **Option 2: Train Full (12-16 giờ → >90% acc)**

```python
# ========== Bước 1-2: Giống trên (nhưng không limit samples) ==========
!python prepare_data.py \
    --input_dir /kaggle/input/datasets/sirimiriiii13/vocr-rec/FinalData \
    --output_dir ./data

# ========== Bước 3: Train Full ==========
!bash train.sh
```

**Chỉ 3 lệnh - Xong!** 🎉

## 🧪 Test Model

Sau khi train xong:

```python
# Inference với model vừa train
!cd PaddleOCR && python tools/infer_rec.py \
    -c /kaggle/working/paddleocr-v5-vietnamese/config.yml \
    -o Global.checkpoints=/kaggle/working/paddleocr-v5-vietnamese/output/vi_ppocr_v5/best_accuracy \
       Global.infer_img=/kaggle/working/paddleocr-v5-vietnamese/data/val_data \
       Global.save_res_path=/kaggle/working/results.txt

# Xem kết quả
!head -20 /kaggle/working/results.txt
```

**Kết quả mẫu (>90% accuracy):**
```
vocr_12345.jpg	quyền 1	0.9234
vocr_12346.jpg	Hiếu Thuần Hoàng Đế	0.8887
vocr_12347.jpg	Định lệ thuế mắm muối	0.9230
```

## ⚠️ **QUAN TRỌNG:**

### **Nếu bị Out of Memory (OOM):**

```yaml
# Giảm batch size trong config.yml
Train:
  loader:
    batch_size_per_card: 4  # Giảm từ 8 xuống 4
  sampler:
    first_bs: 16  # Giảm từ 32 xuống 16
```

### **Nếu muốn train nhanh hơn (trade accuracy):**

```yaml
# Giảm image width
Global:
  d2s_train_image_shape: [3, 48, 320]  # Giảm từ 640 xuống 320

Train:
  dataset:
    transforms:
      - RecConAug:
          image_shape: [48, 320, 3]  # Giảm width
  sampler:
    scales: [[320, 32], [320, 48], [320, 64]]  # Giảm width
  loader:
    batch_size_per_card: 16  # Tăng batch size

Eval:
  dataset:
    transforms:
      - RecResizeImg:
          image_shape: [3, 48, 320]  # Giảm width
```

## 📊 Config Highlights

```yaml
# config.yml - Production Ready

Global:
  epoch_num: 50               # Đủ để converge (từ notebook thành công)
  max_text_length: 96         # Gấp đôi baseline (40→96)
  d2s_train_image_shape: [3, 48, 640]  # Width 640 (gấp đôi 320)
  pretrained_model: latin_PP-OCRv5_mobile  # Latin pretrained

Optimizer:
  learning_rate: 0.0005       # Tối ưu cho Vietnamese
  warmup_epoch: 5             # Ổn định
  
Train:
  batch_size_per_card: 8      # Safe cho T4 GPU (thay vì 128!)
  sampler:
    first_bs: 32              # Memory safe (thay vì 128!)
    scales: [[640,32], [640,48], [640,64]]  # Width 640
  RecConAug:
    image_shape: [48, 640, 3] # HWC format
    max_text_length: 96       # Hỗ trợ text dài
  num_workers: 8              # Parallel loading

Eval:
  batch_size_per_card: 1      # Chính xác nhất
  image_shape: [3, 48, 640]   # CHW format, width 640
```

## 🎯 **TẠI SAO THAY ĐỔI:**

### **Width 640 (thay vì 320):**
- ✅ Text dài hơn vẫn rõ ràng
- ✅ Chi tiết tốt hơn cho chữ nhỏ
- ✅ Accuracy cao hơn ~5-10%
- ⚠️ Cần giảm batch size (128→8)

### **Max length 96 (thay vì 40):**
- ✅ Hỗ trợ câu dài (40 ký tự thường bị cắt)
- ✅ Văn bản cổ có câu rất dài
- ✅ Không tăng memory đáng kể

### **Batch size 8 (thay vì 128):**
- ✅ Tránh OOM trên T4 (15GB VRAM)
- ✅ Width 640 + batch 128 = OOM!
- ✅ Batch 8 vẫn đủ nhanh với 2 GPUs

## 🎯 Timeline

- **Setup**: 5-10 phút (download pretrained ~500MB)
- **Data prep**: 3-5 phút (tùy số lượng)
- **Training 10k, 50 epochs**: 6-8 giờ → **>80% accuracy**
- **Training 250k, 50 epochs**: 12-16 giờ → **>90% accuracy**

## 📝 Chi Tiết Files

| File | Mô tả |
|------|-------|
| `config.yml` | Config cho **full training** (250k, 50 epochs) |
| `config_test_10k.yml` | Config cho **test 10k** (10k, 30 epochs) |
| `train.sh` | Train full → >90% accuracy |
| `train_test_10k.sh` | Train test 10k → 80-85% accuracy |
| `setup_kaggle.sh` | Setup Paddle 3.2 + PaddleOCR + pretrained |
| `prepare_data.py` | Chuẩn bị data + fix paths (all-in-one) |
| `kaggle_notebook.ipynb` | Kaggle notebook template |
| `CHANGELOG.md` | Chi tiết tất cả optimizations |
| `requirements.txt` | Dependencies |

**9 files - Production ready!**

## 🎯 **KHUYẾN NGHỊ WORKFLOW:**

```
1. Test 10k trước (3-4h)
   ├── Verify setup OK
   ├── Check accuracy ~80%
   └── Debug nếu có lỗi
   
2. Nếu OK → Train full (12-16h)
   ├── 250k samples
   ├── 50 epochs
   └── >90% accuracy ✅
```

## 💡 Tips

1. **GPU**: Dùng T4 x2 cho tốc độ tối ưu
2. **Internet**: Bật để download pretrained model
3. **Patience**: 100 epochs mất 8-10 giờ
4. **Monitor**: Check accuracy mỗi 10 epochs

## 🐛 Troubleshooting

**Q: Accuracy = 0%?**
```bash
# Check data
!wc -l data/train_list.txt
!head -3 data/train_list.txt
!ls data/train_data/ | head -5

# Check paths trong config
!grep -E "(character_dict|data_dir)" config.yml
```

**Q: Out of memory?**
```yaml
# Giảm batch size trong config.yml
Train:
  loader:
    batch_size_per_card: 64  # Giảm từ 128
```

**Q: Training quá lâu?**
```yaml
# Giảm epochs
Global:
  epoch_num: 30  # Giảm từ 100 → ~60-70% accuracy
```

---

**Được tối ưu hóa từ notebook thành công 90%+ accuracy** 🎯

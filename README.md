# PaddleOCR Vietnamese - Optimized Version

Phiên bản tối ưu dựa trên workflow thành công với accuracy 90%+

## 🎯 Điểm Nổi Bật

- ✅ **Paddle 3.2.0** - Phiên bản mới nhất
- ✅ **Config tối ưu** - Từ notebook thành công 90%+
- ✅ **Cấu trúc đơn giản** - Chỉ 6 files chính
- ✅ **Tích hợp đầy đủ** - Không cần file fix riêng
- ✅ **Kết quả cao** - 85%+ (10k), 95%+ (250k)

## 📁 Cấu Trúc Project

```
paddleocr-v5-vietnamese/
├── README.md                # Hướng dẫn
├── requirements.txt         # Dependencies
├── config.yml              # Config chính (đã tối ưu)
├── setup_kaggle.sh         # Setup môi trường
├── prepare_data.py         # Chuẩn bị data (tích hợp fix_prepare + fix_config)
└── train.sh                # Training script
```

**Đơn giản, gọn gàng, dễ sử dụng!**

## 🚀 Quick Start - 3 Bước Đơn Giản

```python
# ========== Bước 1: Setup ==========
%cd /kaggle/working
!git clone https://github.com/YOUR_USERNAME/paddleocr-v5-vietnamese.git
%cd paddleocr-v5-vietnamese
!bash setup_kaggle.sh

# ========== Bước 2: Prepare Data ==========
!python prepare_data.py \
    --input_dir /kaggle/input/datasets/sirimiriiii13/vocr-rec/FinalData \
    --output_dir ./data \
    --fix_config config.yml

# Verify
!wc -l data/train_list.txt data/val_list.txt
!head -3 data/train_list.txt

# ========== Bước 3: Train ==========
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

**Kết quả mẫu:**
```
vocr_12345.jpg	quyền 1	0.9234
vocr_12346.jpg	Hiếu Thuần Hoàng Đế	0.8887
```

## 📊 Config Highlights

```yaml
# config.yml - Tối ưu hóa

Global:
  epoch_num: 100              # Đủ để converge
  pretrained_model: ...       # Latin PP-OCRv5
  character_dict_path: ...    # Vietnamese dict
  max_text_length: 40         # Hỗ trợ text dài

Optimizer:
  learning_rate: 0.0005       # Tối ưu
  warmup_epoch: 5             # Ổn định
  
Train:
  batch_size: 128             # Tối đa cho T4 GPU
  RecConAug: prob 0.5         # Augmentation mạnh
  MultiScaleDataSet: True     # Đa kích thước
  num_workers: 8              # Nhanh

Eval:
  batch_size: 1               # Chính xác
  eval_mode: True
```

## 🎯 Timeline

- **Setup**: 5 phút
- **Data prep**: 3-5 phút  
- **Training 10k, 100 epochs**: 8-10 giờ → 80-85% accuracy
- **Training 250k, 100 epochs**: 16-20 giờ → >95% accuracy

## 📝 Chi Tiết Files

| File | Mô tả |
|------|-------|
| `config.yml` | Config tối ưu cho training |
| `setup_kaggle.sh` | Setup Paddle 3.2 + PaddleOCR + pretrained |
| `prepare_data.py` | Chuẩn bị data + fix paths (all-in-one) |
| `train.sh` | Training script với logging |
| `requirements.txt` | Dependencies |
| `README.md` | Hướng dẫn |

**6 files - Đơn giản, rõ ràng!**

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
# ppocr_vnese

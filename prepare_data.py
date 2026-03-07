#!/usr/bin/env python3
"""
Vietnamese OCR Data Preparation Script
Tích hợp tất cả logic từ fix_prepare_data.py và fix_config.py
"""

import os
import sys
import argparse
import random
from pathlib import Path
from tqdm import tqdm
import yaml


def prepare_dataset(input_dir, output_dir, max_samples=None, train_split=0.9):
    """
    Prepare dataset từ FinalData format
    
    Args:
        input_dir: Thư mục chứa images/ và rec_gt.txt
        output_dir: Thư mục output
        max_samples: Giới hạn số samples (None = tất cả)
        train_split: Tỷ lệ train/val (default 0.9)
    """
    print(f"\n{'='*70}")
    print("PREPARING VIETNAMESE OCR DATASET")
    print(f"{'='*70}\n")
    
    # Paths
    images_dir = os.path.join(input_dir, 'images')
    label_file = os.path.join(input_dir, 'rec_gt.txt')
    
    # Verify input
    if not os.path.exists(images_dir):
        print(f"❌ Images directory not found: {images_dir}")
        return False
        
    if not os.path.exists(label_file):
        print(f"❌ Label file not found: {label_file}")
        return False
    
    print(f"📁 Input directory: {input_dir}")
    print(f"📁 Images directory: {images_dir}")
    print(f"📄 Label file: {label_file}\n")
    
    # Create output directories
    os.makedirs(os.path.join(output_dir, 'train_data'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val_data'), exist_ok=True)
    
    # Read labels
    print("📖 Reading labels...")
    with open(label_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    total_lines = len(lines)
    print(f"   Total lines: {total_lines:,}")
    
    # Limit samples if specified
    if max_samples and max_samples < total_lines:
        random.shuffle(lines)
        lines = lines[:max_samples]
        print(f"   Limited to: {max_samples:,} samples")
    
    # Process samples
    print("\n🔄 Processing samples...")
    train_samples = []
    val_samples = []
    valid_count = 0
    missing_count = 0
    
    for line in tqdm(lines, desc="Processing"):
        line = line.strip()
        if not line or '\t' not in line:
            continue
            
        parts = line.split('\t', 1)
        if len(parts) < 2:
            continue
            
        img_path = parts[0]
        text = parts[1]
        
        # FIX: Remove 'images/' prefix if present (fix_prepare_data.py logic)
        if img_path.startswith('images/'):
            img_path = img_path[7:]  # Remove 'images/'
        
        # Full path to source image
        src_img = os.path.join(images_dir, img_path)
        
        # Check if image exists
        if not os.path.exists(src_img):
            missing_count += 1
            continue
        
        # Split train/val
        if random.random() < train_split:
            # Train
            dst_img = os.path.join(output_dir, 'train_data', img_path)
            train_samples.append(f"train_data/{img_path}\t{text}\n")
        else:
            # Val
            dst_img = os.path.join(output_dir, 'val_data', img_path)
            val_samples.append(f"val_data/{img_path}\t{text}\n")
        
        # Create symlink or copy
        os.makedirs(os.path.dirname(dst_img), exist_ok=True)
        if not os.path.exists(dst_img):
            try:
                os.symlink(src_img, dst_img)
            except:
                import shutil
                shutil.copy2(src_img, dst_img)
        
        valid_count += 1
    
    # Save label files
    print("\n💾 Saving label files...")
    
    train_file = os.path.join(output_dir, 'train_list.txt')
    with open(train_file, 'w', encoding='utf-8') as f:
        f.writelines(train_samples)
    print(f"   Train: {train_file} ({len(train_samples):,} samples)")
    
    val_file = os.path.join(output_dir, 'val_list.txt')
    with open(val_file, 'w', encoding='utf-8') as f:
        f.writelines(val_samples)
    print(f"   Val: {val_file} ({len(val_samples):,} samples)")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"✅ Valid samples: {valid_count:,}")
    print(f"   - Train: {len(train_samples):,} ({len(train_samples)/valid_count*100:.1f}%)")
    print(f"   - Val: {len(val_samples):,} ({len(val_samples)/valid_count*100:.1f}%)")
    
    if missing_count > 0:
        print(f"⚠️  Missing images: {missing_count:,}")
    
    print(f"{'='*70}\n")
    
    return True


def _fix_path(old_path, base_dir):
    """Convert relative path or old absolute path to new absolute path"""
    if not old_path:
        return old_path, False
    # Already correct absolute path
    if old_path.startswith(base_dir):
        return old_path, False
    # Old absolute path from different base dir -> extract relative part
    if old_path.startswith('/'):
        # Extract the relative portion after any known base pattern
        for pattern in ['paddleocr-v5-vietnamese/', 'ppocr_vnese/']:
            idx = old_path.find(pattern)
            if idx != -1:
                rel_path = old_path[idx + len(pattern):]
                return os.path.join(base_dir, rel_path), True
        # Unknown absolute path, leave as is
        return old_path, False
    # Relative path -> make absolute
    return os.path.join(base_dir, old_path), True


def fix_config_paths(config_file, base_dir='/kaggle/working/paddleocr-v5-vietnamese'):
    """
    Fix paths trong config file (logic từ fix_config.py)
    Handles both relative paths and old absolute paths from a different base dir.
    
    Args:
        config_file: Path to config YAML file
        base_dir: Base directory for absolute paths
    """
    print(f"\n🔧 Fixing config paths: {config_file}")
    print(f"   Base dir: {base_dir}")
    
    if not os.path.exists(config_file):
        print(f"❌ Config file not found: {config_file}")
        return False
    
    # Read config
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Fix paths
    changed = False
    
    # Global paths
    global_keys = ['character_dict_path', 'pretrained_model', 'save_model_dir', 'save_res_path']
    if 'Global' in config:
        for key in global_keys:
            if key in config['Global'] and config['Global'][key]:
                new_path, did_change = _fix_path(config['Global'][key], base_dir)
                if did_change:
                    config['Global'][key] = new_path
                    print(f"   Fixed: {key}")
                    changed = True
    
    # Train dataset paths
    if 'Train' in config and 'dataset' in config['Train']:
        ds = config['Train']['dataset']
        if 'data_dir' in ds:
            new_path, did_change = _fix_path(ds['data_dir'], base_dir)
            if did_change:
                ds['data_dir'] = new_path
                print(f"   Fixed: Train data_dir")
                changed = True
        if 'label_file_list' in ds:
            new_list = []
            for old_path in ds['label_file_list']:
                new_path, did_change = _fix_path(old_path, base_dir)
                new_list.append(new_path)
                if did_change:
                    changed = True
            ds['label_file_list'] = new_list
            if changed:
                print(f"   Fixed: Train label_file_list")
    
    # Eval dataset paths
    if 'Eval' in config and 'dataset' in config['Eval']:
        ds = config['Eval']['dataset']
        if 'data_dir' in ds:
            new_path, did_change = _fix_path(ds['data_dir'], base_dir)
            if did_change:
                ds['data_dir'] = new_path
                print(f"   Fixed: Eval data_dir")
                changed = True
        if 'label_file_list' in ds:
            new_list = []
            for old_path in ds['label_file_list']:
                new_path, did_change = _fix_path(old_path, base_dir)
                new_list.append(new_path)
                if did_change:
                    changed = True
            ds['label_file_list'] = new_list
            if changed:
                print(f"   Fixed: Eval label_file_list")
    
    # Save if changed
    if changed:
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        print(f"✅ Config updated: {config_file}\n")
    else:
        print(f"✓ Config already has absolute paths\n")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Prepare Vietnamese OCR dataset')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory containing images/ and rec_gt.txt')
    parser.add_argument('--output_dir', type=str, default='./data',
                        help='Output directory (default: ./data)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to use')
    parser.add_argument('--train_split', type=float, default=0.9,
                        help='Train/val split ratio (default: 0.9)')
    parser.add_argument('--fix_config', type=str, default=None,
                        help='Config file to fix paths (optional)')
    parser.add_argument('--base_dir', type=str, default='/kaggle/working/paddleocr-v5-vietnamese',
                        help='Base directory for config paths')
    
    args = parser.parse_args()
    
    # Fix config if specified (run independently of data prep)
    if args.fix_config:
        fix_config_paths(args.fix_config, args.base_dir)
    
    # Prepare dataset (skip if input_dir is /dev/null or doesn't exist)
    if args.input_dir == '/dev/null':
        print("\n⏭️  Skipping data preparation (fix_config only mode)")
    else:
        success = prepare_dataset(
            args.input_dir,
            args.output_dir,
            max_samples=args.max_samples,
            train_split=args.train_split
        )
        if not success:
            sys.exit(1)
    
    print("✅ All done!\n")


if __name__ == '__main__':
    main()

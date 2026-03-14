#!/usr/bin/env python3
"""
Vietnamese OCR Data Preparation Script
- Prepares dataset from FinalData format
- AUTO-GENERATES dictionary from actual training data
- Fixes config paths and out_channels_list

Usage:
    python prepare_data.py --input_dir /path/to/FinalData --output_dir ./data --fix_config config.yml
"""

import os
import sys
import argparse
import random
import unicodedata
from pathlib import Path
from tqdm import tqdm
import yaml


def prepare_dataset(input_dir, output_dir, max_samples=None, train_split=0.9):
    """
    Prepare dataset from FinalData format
    
    Args:
        input_dir: Directory containing images/ and rec_gt.txt
        output_dir: Output directory
        max_samples: Limit number of samples (None = all)
        train_split: Train/val split ratio
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
    
    # Read all labels
    print("📖 Reading labels...")
    samples = []
    
    with open(label_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t')
            if len(parts) < 2:
                print(f"⚠️  Line {line_num}: Bad format (no tab separator)")
                continue
            
            # Remove 'images/' prefix if exists
            img_name = parts[0].replace('images/', '').strip()
            text = parts[1].strip()
            
            # Normalize text to NFC
            text = unicodedata.normalize('NFC', text)
            
            # Check image exists
            img_path = os.path.join(images_dir, img_name)
            if not os.path.exists(img_path):
                if line_num <= 10:  # Only warn for first 10
                    print(f"⚠️  Line {line_num}: Image not found: {img_name}")
                continue
            
            samples.append({
                'image_path': img_path,
                'image_name': img_name,
                'text': text
            })
    
    print(f"✅ Found {len(samples)} valid samples\n")
    
    if len(samples) == 0:
        print("❌ No valid samples found!")
        return False
    
    # Limit samples if requested
    if max_samples and len(samples) > max_samples:
        print(f"🎯 Limiting to {max_samples} samples")
        random.shuffle(samples)
        samples = samples[:max_samples]
    
    # Split train/val
    random.shuffle(samples)
    split_idx = int(len(samples) * train_split)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]
    
    print(f"📊 Split: {len(train_samples)} train, {len(val_samples)} val\n")
    
    # Copy files and create lists
    print("💾 Copying images and creating lists...")
    
    train_list = []
    for i, sample in enumerate(tqdm(train_samples, desc="Train")):
        # New filename
        ext = os.path.splitext(sample['image_name'])[1]
        new_name = f"train_{i:06d}{ext}"
        dest = os.path.join(output_dir, 'train_data', new_name)
        
        # Copy image
        import shutil
        shutil.copy2(sample['image_path'], dest)
        
        # Add to list
        train_list.append(f"train_data/{new_name}\t{sample['text']}\n")
    
    val_list = []
    for i, sample in enumerate(tqdm(val_samples, desc="Val")):
        ext = os.path.splitext(sample['image_name'])[1]
        new_name = f"val_{i:06d}{ext}"
        dest = os.path.join(output_dir, 'val_data', new_name)
        
        import shutil
        shutil.copy2(sample['image_path'], dest)
        
        val_list.append(f"val_data/{new_name}\t{sample['text']}\n")
    
    # Save lists
    train_list_path = os.path.join(output_dir, 'train_list.txt')
    val_list_path = os.path.join(output_dir, 'val_list.txt')
    
    with open(train_list_path, 'w', encoding='utf-8') as f:
        f.writelines(train_list)
    
    with open(val_list_path, 'w', encoding='utf-8') as f:
        f.writelines(val_list)
    
    print(f"\n✅ Dataset prepared:")
    print(f"   Train list: {train_list_path} ({len(train_list)} samples)")
    print(f"   Val list: {val_list_path} ({len(val_list)} samples)")
    print(f"   Train images: {os.path.join(output_dir, 'train_data/')}")
    print(f"   Val images: {os.path.join(output_dir, 'val_data/')}\n")
    
    return True


def generate_dictionary_from_data(data_dir, dict_path):
    """
    Generate dictionary from actual training data
    This ensures ALL characters in data are in dictionary
    
    Args:
        data_dir: Directory containing train_list.txt
        dict_path: Path to save dictionary
        
    Returns:
        Number of characters in dictionary
    """
    print(f"\n{'='*70}")
    print("AUTO-GENERATING DICTIONARY FROM DATA")
    print(f"{'='*70}\n")
    
    train_list = os.path.join(data_dir, 'train_list.txt')
    
    if not os.path.exists(train_list):
        print(f"❌ {train_list} not found")
        return None
    
    # Collect all unique characters
    all_chars = set()
    sample_count = 0
    
    print("📖 Scanning training data for characters...")
    with open(train_list, 'r', encoding='utf-8') as f:
        for line in f:
            if '\t' in line:
                text = line.split('\t', 1)[1].strip()
                # Normalize to NFC
                text_normalized = unicodedata.normalize('NFC', text)
                all_chars.update(text_normalized)
                sample_count += 1
    
    # Sort for consistency
    chars_sorted = sorted(all_chars)
    
    print(f"✅ Analyzed {sample_count} samples")
    print(f"✅ Found {len(chars_sorted)} unique characters\n")
    print(f"📝 First 30 chars: {chars_sorted[:30]}")
    print(f"📝 Last 30 chars: {chars_sorted[-30:]}\n")
    
    # Save dictionary - 1 character per line (CRITICAL!)
    os.makedirs(os.path.dirname(dict_path), exist_ok=True)
    
    with open(dict_path, 'w', encoding='utf-8') as f:
        for char in chars_sorted:
            f.write(char + '\n')
    
    print(f"✅ Dictionary saved: {dict_path}")
    print(f"   Total characters: {len(chars_sorted)}")
    print(f"   Format: 1 character per line\n")
    
    # Verify
    with open(dict_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"📋 Verification:")
    print(f"   Lines in file: {len(lines)}")
    print(f"   First 10 chars:")
    for i, line in enumerate(lines[:10]):
        char = line.strip()
        print(f"      {i+1}. '{char}' (len={len(char)})")
    
    return len(chars_sorted)


def fix_config_paths(config_file, base_dir, char_num=None, dict_path=None):
    """
    Fix paths in config and update out_channels_list
    
    Args:
        config_file: Path to config YAML
        base_dir: Base directory for absolute paths
        char_num: Number of characters (auto-updates out_channels_list)
        dict_path: Dictionary path (auto-updates character_dict_path)
    """
    print(f"\n{'='*70}")
    print("FIXING CONFIG")
    print(f"{'='*70}\n")
    
    if not os.path.exists(config_file):
        print(f"❌ Config not found: {config_file}")
        return False
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    changed = False
    
    # Update dictionary path
    if dict_path and 'Global' in config:
        config['Global']['character_dict_path'] = dict_path
        print(f"✅ Updated character_dict_path: {dict_path}")
        changed = True
    
    # Update out_channels_list (CRITICAL!)
    if char_num and 'Architecture' in config and 'Head' in config['Architecture']:
        config['Architecture']['Head']['out_channels_list'] = {
            'CTCLabelDecode': char_num,
            'NRTRLabelDecode': char_num + 3  # +3 for special tokens
        }
        print(f"✅ Updated out_channels_list:")
        print(f"   CTCLabelDecode: {char_num}")
        print(f"   NRTRLabelDecode: {char_num + 3}")
        changed = True
    
    # Fix other paths to absolute
    if 'Global' in config:
        for key in ['pretrained_model', 'save_model_dir']:
            if key in config['Global'] and config['Global'][key]:
                old_path = config['Global'][key]
                if not os.path.isabs(old_path):
                    config['Global'][key] = os.path.join(base_dir, old_path)
                    print(f"   Fixed: {key}")
                    changed = True
    
    if 'Train' in config and 'dataset' in config['Train']:
        if 'data_dir' in config['Train']['dataset']:
            old = config['Train']['dataset']['data_dir']
            if not os.path.isabs(old):
                config['Train']['dataset']['data_dir'] = os.path.join(base_dir, old)
                changed = True
        
        if 'label_file_list' in config['Train']['dataset']:
            new_list = []
            for old in config['Train']['dataset']['label_file_list']:
                new_list.append(os.path.join(base_dir, old) if not os.path.isabs(old) else old)
            config['Train']['dataset']['label_file_list'] = new_list
            changed = True
    
    if 'Eval' in config and 'dataset' in config['Eval']:
        if 'data_dir' in config['Eval']['dataset']:
            old = config['Eval']['dataset']['data_dir']
            if not os.path.isabs(old):
                config['Eval']['dataset']['data_dir'] = os.path.join(base_dir, old)
                changed = True
        
        if 'label_file_list' in config['Eval']['dataset']:
            new_list = []
            for old in config['Eval']['dataset']['label_file_list']:
                new_list.append(os.path.join(base_dir, old) if not os.path.isabs(old) else old)
            config['Eval']['dataset']['label_file_list'] = new_list
            changed = True
    
    if changed:
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
        print(f"\n✅ Config updated: {config_file}\n")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Prepare Vietnamese OCR dataset with auto dictionary generation'
    )
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory (contains images/ and rec_gt.txt)')
    parser.add_argument('--output_dir', type=str, default='./data',
                        help='Output directory (default: ./data)')
    parser.add_argument('--dict_path', type=str, default='./dict/vi_dict.txt',
                        help='Dictionary output path (default: ./dict/vi_dict.txt)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Limit number of samples (default: all)')
    parser.add_argument('--train_split', type=float, default=0.9,
                        help='Train/val split ratio (default: 0.9)')
    parser.add_argument('--fix_config', type=str, default=None,
                        help='Config file to update (optional)')
    parser.add_argument('--base_dir', type=str, default='/kaggle/working/ppocr_vnese',
                        help='Base directory for config paths')
    
    args = parser.parse_args()
    
    # Step 1: Prepare dataset
    success = prepare_dataset(
        args.input_dir,
        args.output_dir,
        max_samples=args.max_samples,
        train_split=args.train_split
    )
    
    if not success:
        print("\n❌ Dataset preparation failed!")
        sys.exit(1)
    
    # Step 2: Generate dictionary from data (AUTO!)
    char_num = generate_dictionary_from_data(args.output_dir, args.dict_path)
    
    if not char_num:
        print("\n❌ Dictionary generation failed!")
        sys.exit(1)
    
    # Step 3: Fix config if specified
    if args.fix_config:
        fix_config_paths(args.fix_config, args.base_dir, char_num, args.dict_path)
    
    print(f"{'='*70}")
    print("✅ ALL DONE!")
    print(f"{'='*70}")
    print(f"Dataset: {args.output_dir}")
    print(f"Dictionary: {args.dict_path} ({char_num} chars)")
    if args.fix_config:
        print(f"Config: {args.fix_config} (updated)")
    print()


if __name__ == '__main__':
    main()

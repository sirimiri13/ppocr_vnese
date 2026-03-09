import paddle
import yaml
import cv2
import numpy as np
from paddle.nn import functional as F
from ctc_beam_search_decode import CTCBeamSearchDecode


def load_config(config_path='config.yml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def preprocess(img_path, shape=[3, 48, 320]):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Không tìm thấy ảnh: {img_path}")
    img = cv2.resize(img, (shape[2], shape[1]))
    img = img.astype('float32') / 255.0
    img -= 0.5
    img /= 0.5
    img = img.transpose((2, 0, 1))  # HWC -> CHW
    return img[np.newaxis, :]  # add batch dim


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Đường dẫn ảnh cần nhận dạng')
    parser.add_argument('--config', type=str, default='config.yml')
    parser.add_argument('--model_dir', type=str, default='output/vi_ppocr_v5/best_accuracy')
    parser.add_argument('--beam_width', type=int, default=10)
    parser.add_argument('--k_best', type=int, default=5)
    args = parser.parse_args()

    config = load_config(args.config)

    # 1. Build & load model
    from paddleocr.ppocr.modeling.architectures import build_model
    from paddleocr.ppocr.utils.save_load import load_model

    model = build_model(config['Architecture'])
    load_model(config, model)
    model.eval()

    # 2. Beam search decoder (thay thế CTCLabelDecode)
    decoder = CTCBeamSearchDecode(
        character_dict_path=config['Global']['character_dict_path'],
        use_space_char=config['Global'].get('use_space_char', True),
        beam_width=args.beam_width,
        k_best=args.k_best
    )

    # 3. Preprocess
    img_shape = config['Global'].get('d2s_train_image_shape', [3, 48, 320])
    img = preprocess(args.image, shape=img_shape)
    input_tensor = paddle.to_tensor(img)

    # 4. Forward
    with paddle.no_grad():
        preds = model(input_tensor)

    # 5. Decode k-best
    k_results = decoder(preds)

    print("=" * 50)
    print(f"Ảnh: {args.image}")
    print(f"Beam width: {args.beam_width}, K-best: {args.k_best}")
    print("-" * 50)
    for rank, (text, score) in enumerate(k_results[0]):
        print(f"  Top-{rank+1}: '{text}' (score: {score:.6f})")
    print("=" * 50)


if __name__ == '__main__':
    main()

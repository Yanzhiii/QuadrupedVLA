#!/usr/bin/env python3
"""
Offline NaVILA test script: No robot required, no ROS required.
Use local images or random image sequences + instructions, run inference directly and print results.

Usage:
  conda activate navila
  python tools/offline_navila_eval.py --images path/to/dir --frames 8 --instr "move forward"
  Or without --images, will use random images.
"""
import argparse
import os
from pathlib import Path
from PIL import Image
import numpy as np

import sys
WS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS_DIR / 'src' / 'qvla_bridge'))

# Allow llava to be imported from source
os.environ.setdefault('PYTHONPATH', '')
os.environ['PYTHONPATH'] = f"{WS_DIR / 'src' / 'navila_repo_main'}:{os.environ['PYTHONPATH']}"

from qvla_bridge.model_adapters import call_navila_inference


def load_images_from_dir(directory: Path, target_frames: int):
    images = []
    if directory and directory.exists():
        for p in sorted(directory.glob('*.jpg')) + sorted(directory.glob('*.png')):
            try:
                images.append(Image.open(p).convert('RGB'))
                if len(images) >= target_frames:
                    break
            except Exception:
                pass
    return images


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', type=str, default='', help='Image directory, can be empty')
    ap.add_argument('--frames', type=int, default=8, help='Number of frames to use')
    ap.add_argument('--instr', type=str, default='move forward', help='Instruction')
    args = ap.parse_args()

    if args.images:
        imgs = load_images_from_dir(Path(args.images), args.frames)
    else:
        imgs = []

    if len(imgs) < args.frames:
        # Fill with random images if insufficient
        for _ in range(args.frames - len(imgs)):
            arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            imgs.append(Image.fromarray(arr))

    print(f"Frames: {len(imgs)}, Instr: {args.instr}")
    out = call_navila_inference(imgs, args.instr)
    print('Output:', out)


if __name__ == '__main__':
    main()



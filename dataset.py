"""
dataset.py
────────────────────────────────────────────────────────────────────────────────
MultiDataset now loads three modalities:
  • text   — from item["text"]  (BERT tokens)
  • video  — from item["video"] (single mid-frame, ImageNet-normalised)
  • audio  — derived from item["video"] path:
               datasets/videos/fake/X.mp4  →  datasets/audios/fake/X.wav
               datasets/videos/real/X.mp4  →  datasets/audios/real/X.wav
             Falls back to zero MFCC vector if WAV is absent.

Label priority:
  1. item["label"]  if present (backward-compat with old dataset.json)
  2. folder name of the video path  (fake=1 / real=0)
  3. folder name of the audio path
────────────────────────────────────────────────────────────────────────────────
"""

import os
import torch
from torch.utils.data import Dataset
import json

from preprocessing.text  import process_text
from preprocessing.video import extract_frames, label_from_video_path
from preprocessing.audio_features import extract_audio_features, label_from_audio_path

def _derive_audio_path(video_path: str) -> str:
    """
    Swap  datasets/videos/<split>/<name>.ext
    →     datasets/audios/<split>/<name>.wav
    """
    if video_path is None:
        return None
    norm = video_path.replace("\\", "/")
    parts = norm.split("/")
    try:
        vid_idx = parts.index("videos")
        parts[vid_idx] = "audios"
    except ValueError:
        pass   # path doesn't contain "videos" — leave as-is, WAV will be missing
    base   = os.path.splitext("/".join(parts))[0]
    return base + ".wav"


class MultiDataset(Dataset):
    def __init__(self, path: str):
        with open(path) as f:
            self.samples = json.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        # ── text ──────────────────────────────────────────────────────────────
        t = process_text(item["text"])

        # ── video ─────────────────────────────────────────────────────────────
        video_path = item.get("video")
        v = extract_frames(video_path)
        assert v.shape == (224, 224, 3), f"Unexpected frame shape {v.shape}"

        # ── audio  (derived from video path) ──────────────────────────────────
        audio_path = item.get("audio") or _derive_audio_path(video_path)
        a = extract_audio_features(audio_path) if audio_path else np.zeros(40, dtype=np.float32)   # (40,)

        # ── label ─────────────────────────────────────────────────────────────
        if "label" in item:
            label = int(item["label"])
        else:
            label = label_from_video_path(video_path)
            if label == -1:
                label = label_from_audio_path(audio_path)

        return {
            "input_ids":      t["input_ids"].squeeze(0),                         # (128,)
            "attention_mask": t["attention_mask"].squeeze(0),                    # (128,)
            "video":          torch.tensor(v, dtype=torch.float32).permute(2, 0, 1),  # (3,224,224)
            "audio":          torch.tensor(a, dtype=torch.float32),              # (40,)
            "label":          torch.tensor(label, dtype=torch.long),
            # ── metadata for UI demo ──────────────────────────────────────────
            "video_path":     video_path or "",
            "audio_path":     audio_path or "",
            "video_label":    label_from_video_path(video_path),
            "audio_label":    label_from_audio_path(audio_path),
        }
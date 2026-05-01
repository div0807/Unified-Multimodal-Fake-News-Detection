"""
inference.py
───────────────────────────────────────────────────────────────────────────────
Three-modality inference: Text + Video + Audio.

• Text  : BERT-base-uncased → 256-dim projection
• Video : ResNet-18 mid-frame → 256-dim projection
• Audio : Mean MFCCs (40-dim) → 256-dim MLP
• Fusion: FusionModelV3  (cross-modal attention)

Graceful fallbacks:
  - No video  → zero frame
  - No audio  → zero MFCC vector (40-dim)
  - No text   → empty string (tokenises to padding)
───────────────────────────────────────────────────────────────────────────────
"""
import os
import torch
import numpy as np
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from models.audio_svm import predict_audio_svm


from models.text_model          import TextModel
from models.video_model         import VideoModel

from models.attention_fusion_v3 import FusionModelV3

from preprocessing.text  import process_text
from preprocessing.video import extract_frames



# ───────────────────────────────────────────────────────────────────────────────
# Device
# ───────────────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_models = None


# ───────────────────────────────────────────────────────────────────────────────
# Load models (cached)
# ───────────────────────────────────────────────────────────────────────────────
def load_models():
    global _models
    if _models is not None:
        return _models

    ckpt = torch.load("best_model.pth", map_location=device)

    tm = TextModel().to(device)
    vm = VideoModel().to(device)
    fm = FusionModelV3().to(device)

    tm.load_state_dict(ckpt["tm"])
    vm.load_state_dict(ckpt["vm"])
    fm.load_state_dict(ckpt["fm"])

    tm.eval()
    vm.eval()
    fm.eval()

    _models = (tm, vm, fm)
    return _models
# ───────────────────────────────────────────────────────────────────────────────
# Predict
# ───────────────────────────────────────────────────────────────────────────────
def predict(text: str, video_path: str = None, audio_path: str = None) -> dict:
    """
    Run inference on any combination of modalities.
    """

    tm, vm, fm = load_models()
    with torch.no_grad():

        # ── TEXT ────────────────────────────────────────────────────────────
        t = process_text(text or "")
        input_ids = t["input_ids"].to(device)
        attention_mask = t["attention_mask"].to(device)

        t_out = tm(input_ids, attention_mask)   # (1,256)

        # ── VIDEO ───────────────────────────────────────────────────────────
        if video_path and os.path.exists(video_path):
            frame = extract_frames(video_path)  # (224,224,3)
        else:
            # zero-frame fallback
            frame = np.zeros((224, 224, 3), dtype=np.float32)

        v_tensor = (
            torch.tensor(frame, dtype=torch.float32)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device)
        )

        v_out = vm(v_tensor)  # (1,256)

        # ── AUDIO ───────────────────────────────────────────────────────────
        # Auto-derive audio path from video path if not supplied
        if audio_path is None and video_path:
            parts = video_path.replace("\\", "/").split("/")
            try:
                parts[parts.index("videos")] = "audios"
                candidate = os.path.splitext("/".join(parts))[0] + ".wav"
                if os.path.exists(candidate):
                    audio_path = candidate
            except ValueError:
                pass

        if audio_path and os.path.exists(audio_path):
            score = predict_audio_svm(audio_path)
            a_out = torch.ones((1,256)).to(device) * score
        else:
            a_out = torch.zeros((1,256)).to(device)

        # ── FUSION ──────────────────────────────────────────────────────────
        logits = fm(t_out, v_out, a_out)
        probs = torch.softmax(logits, dim=1)[0]

        if not (text and text.strip()):
            probs = probs * 0.7 + 0.3 * torch.tensor([0.5, 0.5]).to(probs.device)

    # ── Modality tag ────────────────────────────────────────────────────────
    has_text = bool(text and text.strip())
    has_video = bool(video_path)
    has_audio = bool(audio_path)

    if has_text and has_video and has_audio:
        modality = "Text + Video + Audio"
    elif has_text and has_video:
        modality = "Text + Video"
    elif has_text and has_audio:
        modality = "Text + Audio"
    elif has_video and has_audio:
        modality = "Video + Audio"
    elif has_text:
        modality = "Text only"
    elif has_video:
        modality = "Video only"
    elif has_audio:
        modality = "Audio only"
    else:
        modality = "Fallback (no input)"

    return {
        "label":      "FAKE" if probs[1] > probs[0] else "REAL",
        "fake_prob":  float(probs[1]),
        "real_prob":  float(probs[0]),
        "confidence": float(probs.max()),
        "modality":   modality,
    }
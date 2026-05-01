import cv2
import numpy as np


# ImageNet normalization constants (what ResNet-18 was trained on)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def extract_frames(video_path: str, size: int = 224) -> np.ndarray:
    """
    Extract a single representative frame from a video file.

    Returns:
        np.ndarray of shape (224, 224, 3), dtype float32, ImageNet-normalized.
        Falls back to a zero array if the video cannot be read.
    """
    fallback = np.zeros((size, size, 3), dtype=np.float32)

    if video_path is None:
        return fallback

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return fallback

    # Seek to the middle frame for a more representative sample
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mid = max(0, total_frames // 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid)

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return fallback

    # OpenCV reads BGR → convert to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize to target size
    frame = cv2.resize(frame, (size, size))

    # Convert to float32 in [0, 1]
    frame = frame.astype(np.float32) / 255.0

    # ImageNet normalization
    frame = (frame - _MEAN) / _STD

    return frame  # (224, 224, 3)

def label_from_video_path(video_path: str) -> int:
    if video_path is None:
        return -1
    parts = video_path.replace("\\", "/").split("/")
    for p in parts:
        p = p.lower()
        if p == "fake":
            return 1
        if p == "real":
            return 0
    return -1
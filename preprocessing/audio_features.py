"""
preprocessing/audio.py
────────────────────────────────────────────────────────────────────────────────
Extract mean MFCC features (40-dim) from a .wav audio file using scipy only
(no librosa dependency).

Label convention (mirrors video folder convention):
  datasets/audios/fake/<file>.wav  →  label = 1
  datasets/audios/real/<file>.wav  →  label = 0
  None / unreadable                →  zero vector, label = -1 (ignored in loss)
────────────────────────────────────────────────────────────────────────────────
"""
from scipy.fft import dct
import os
import wave
import struct
import numpy as np
from scipy.fft import dct

# ── constants ──────────────────────────────────────────────────────────────────
N_MFCC      = 40
N_FILTERS   = 40
FFT_SIZE    = 512
HOP_SIZE    = 160   # ~10 ms at 16 kHz
WIN_SIZE    = 400   # ~25 ms at 16 kHz
SAMPLE_RATE = 16_000


# ── helpers ────────────────────────────────────────────────────────────────────

def _read_wav(path: str):
    """Read a mono/stereo WAV → mono float32 array at original sample rate."""
    with wave.open(path, "rb") as wf:
        sr      = wf.getframerate()
        n_ch    = wf.getnchannels()
        sw      = wf.getsampwidth()
        n_frame = wf.getnframes()
        raw     = wf.readframes(n_frame)

    fmt = {1: "b", 2: "h", 4: "i"}.get(sw, "h")
    samples = np.array(struct.unpack(f"{n_frame * n_ch}{fmt}", raw), dtype=np.float32)
    samples /= float(2 ** (sw * 8 - 1))   # normalise to [-1, 1]

    if n_ch > 1:
        samples = samples.reshape(-1, n_ch).mean(axis=1)   # stereo → mono

    return samples, sr


def _resample_naive(sig: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Linear interpolation resample (good enough for MFCC features)."""
    if orig_sr == target_sr:
        return sig
    duration  = len(sig) / orig_sr
    new_len   = int(duration * target_sr)
    x_old     = np.linspace(0, duration, len(sig))
    x_new     = np.linspace(0, duration, new_len)
    return np.interp(x_new, x_old, sig).astype(np.float32)


def _mel_filterbank(sr: int, n_fft: int, n_filters: int,
                    fmin: float = 0.0, fmax: float = None) -> np.ndarray:
    """Create a (n_filters, n_fft//2+1) mel filterbank matrix."""
    if fmax is None:
        fmax = sr / 2.0

    def hz_to_mel(f):  return 2595 * np.log10(1 + f / 700)
    def mel_to_hz(m):  return 700 * (10 ** (m / 2595) - 1)

    mel_low  = hz_to_mel(fmin)
    mel_high = hz_to_mel(fmax)
    mels     = np.linspace(mel_low, mel_high, n_filters + 2)
    hz_pts   = mel_to_hz(mels)
    bin_pts  = np.floor((n_fft + 1) * hz_pts / sr).astype(int)

    fb = np.zeros((n_filters, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_filters + 1):
        lo, ctr, hi = bin_pts[m - 1], bin_pts[m], bin_pts[m + 1]
        for k in range(lo, ctr):
            if ctr != lo:
                fb[m - 1, k] = (k - lo) / (ctr - lo)
        for k in range(ctr, hi):
            if hi != ctr:
                fb[m - 1, k] = (hi - k) / (hi - ctr)
    return fb


def _compute_mfcc(sig: np.ndarray, sr: int,
                  n_mfcc: int = N_MFCC,
                  n_filters: int = N_FILTERS,
                  n_fft: int = FFT_SIZE,
                  hop: int = HOP_SIZE,
                  win: int = WIN_SIZE) -> np.ndarray:
    """Return mean MFCCs over time, shape (n_mfcc,)."""
    # Pre-emphasis
    sig = np.append(sig[0], sig[1:] - 0.97 * sig[:-1])

    # Frame the signal
    n_frames = max(1, (len(sig) - win) // hop + 1)
    frames   = np.stack([sig[i * hop: i * hop + win] for i in range(n_frames)])

    # Window
    window = np.hanning(win).astype(np.float32)
    frames = frames[:, :win] * window

    # Power spectrum
    mag    = np.abs(np.fft.rfft(frames, n=n_fft)) ** 2   # (T, n_fft//2+1)

    # Mel filterbank
    fb     = _mel_filterbank(sr, n_fft, n_filters)        # (n_filters, n_fft//2+1)
    mel_e  = mag @ fb.T                                    # (T, n_filters)
    mel_e  = np.where(mel_e == 0, np.finfo(float).eps, mel_e)
    log_me = np.log(mel_e)

    # DCT → MFCCs
    mfcc   = dct(log_me, type=2, axis=1, norm="ortho")[:, :n_mfcc]   # (T, n_mfcc)

    return mfcc.mean(axis=0).astype(np.float32)   # (n_mfcc,)


# ── public API ─────────────────────────────────────────────────────────────────

def extract_audio_features(audio_path: str, n_mfcc: int = N_MFCC) -> np.ndarray:
    """
    Extract mean MFCC features from a WAV file.

    Returns:
        np.ndarray of shape (n_mfcc,) = (40,), dtype float32.
        Falls back to zeros if the file cannot be read.
    """
    fallback = np.zeros(n_mfcc, dtype=np.float32)
    if audio_path is None:
        return fallback
    try:
        sig, sr = _read_wav(audio_path)
        sig     = _resample_naive(sig, sr, SAMPLE_RATE)
        return  _compute_mfcc(sig, SAMPLE_RATE, n_mfcc=n_mfcc)
    except Exception:
        return fallback


def label_from_audio_path(audio_path: str) -> int:
    """
    Derive ground-truth label from folder convention:
      datasets/audios/fake/...  → 1
      datasets/audios/real/...  → 0
      anything else             → -1  (ignored during training)
    """
    if audio_path is None:
        return -1
    parts = audio_path.replace("\\", "/").split("/")
    for part in parts:
        p = part.lower()
        if p == "fake":
            return 1
        if p == "real":
            return 0
    return -1
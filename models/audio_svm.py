import librosa
import numpy as np
import joblib
import os

MODEL_PATH = "svm_model.pkl"
SCALER_PATH = "scaler.pkl"

def extract_mfcc(audio_path, n_mfcc=13):
 audio, sr = librosa.load(audio_path, sr=None)
 mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
 mfcc_mean = np.mean(mfcc.T, axis=0)
 mfcc_std  = np.std(mfcc.T, axis=0)
 return np.hstack((mfcc_mean, mfcc_std))  # 26 features

def predict_audio_svm(audio_path):
 if not os.path.exists(audio_path):
  return 0.5

 mfcc = extract_mfcc(audio_path)

 scaler = joblib.load(SCALER_PATH)
 model = joblib.load(MODEL_PATH)

 mfcc_scaled = scaler.transform(mfcc.reshape(1, -1))
 pred = model.predict(mfcc_scaled)[0]

 return 0.8 if pred == 1 else 0.2
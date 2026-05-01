import os
import glob
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

def extract_mfcc(audio_path):
 audio, sr = librosa.load(audio_path, sr=16000)

 max_len = 3 * sr
 if len(audio) > max_len:
  audio = audio[:max_len]
 else:
  audio = np.pad(audio, (0, max_len - len(audio)))

 mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
 mfcc_mean = np.mean(mfcc.T, axis=0)
 mfcc_std  = np.std(mfcc.T, axis=0)
 return np.hstack((mfcc_mean, mfcc_std))

def load_data(folder, label):
 X, y = [], []
 for file in glob.glob(folder + "/*.wav"):
  feat = extract_mfcc(file)
  X.append(feat)
  y.append(label)
 return X, y

def main():
 real_X, real_y = load_data("real_audio", 0)
 fake_X, fake_y = load_data("deepfake_audio", 1)

 print("Real:", len(real_X), "Fake:", len(fake_X))

 X = np.vstack((real_X, fake_X))
 y = np.hstack((real_y, fake_y))

 scaler = StandardScaler()
 X_scaled = scaler.fit_transform(X)
 model = SVC(kernel="rbf", C=10, gamma="scale", probability=True)
 model.fit(X_scaled, y)

 joblib.dump(model, "svm_model.pkl")
 joblib.dump(scaler, "scaler.pkl")

 print("✅ Audio model trained and saved")

if __name__ == "__main__":
 main()
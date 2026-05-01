from models.audio_svm import predict_audio_svm

print("Real:", predict_audio_svm("real_audio/Furqanreal.wav"))
print("Fake:", predict_audio_svm("deepfake_audio/FurqanAIClone.wav"))
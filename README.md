\# 🧠 Unified Multimodal Fake News Detection



\## 📌 Overview



This project presents a \*\*multimodal fake news detection system\*\* that analyzes \*\*text, video, and audio\*\* simultaneously to classify content as \*\*REAL or FAKE\*\*.

By combining multiple modalities, the system improves reliability compared to single-source (unimodal) approaches.



\---



\## 🚀 Features



\* 🔤 \*\*Text Analysis\*\* using BERT-based embeddings

\* 🎥 \*\*Video Analysis\*\* using ResNet-18

\* 🔊 \*\*Audio Analysis\*\* using MFCC + SVM

\* 🔗 \*\*Multimodal Fusion\*\* using attention-based model

\* ⚡ Real-time inference via Streamlit UI

\* 📊 Evaluation with Accuracy, Precision, Recall, F1 Score



\---



\## 🏗️ Project Structure



```

├── datasets/

├── models/

│   ├── text\_model.py

│   ├── video\_model.py

│   ├── audio\_model.py

│   ├── attention\_fusion\_v3.py

├── preprocessing/

│   ├── text.py

│   ├── video.py

│   ├── audio\_features.py

├── app.py

├── train.py

├── inference.py

├── evaluate.py

├── dataset.json

├── requirements.txt

```



\---



\## ⚙️ Installation



\### 1. Install dependencies



```

pip install -r requirements.txt

```



\---



\## 🏋️ Training



Train the multimodal model:



```

python train.py

```



This will generate:



```

best\_model.pth

```



\---



\## 🔊 Train Audio Model (SVM)



```

python train\_audio\_svm.py

```



This generates:



```

svm\_model.pkl

scaler.pkl

```



\---



\## 🧪 Evaluation



Run evaluation metrics and graphs:



```

python evaluate.py

```



Outputs:



\* Confusion Matrix

\* Accuracy / Precision / Recall / F1

\* Performance graphs



\---



\## 🎯 Inference (Prediction)



Run the Streamlit app:



```

streamlit run app.py

```



Upload:



\* Text (optional)

\* Video (optional)

\* Audio (optional)



The system predicts:



```

REAL / FAKE

Confidence Score

```



\---



\## 📊 Results



\* \*\*Accuracy:\*\* \~82%

\* \*\*Precision:\*\* \~76%

\* \*\*Recall:\*\* \~52%

\* \*\*F1 Score:\*\* \~62%



\---



\## 🧠 Methodology



1\. Data Collection (Text, Video, Audio)

2\. Preprocessing

3\. Train–Validation–Test Split

4\. Feature Extraction

5\. Feature Fusion

6\. Model Training

7\. Evaluation



\---



\## ⚠️ Challenges



\* Limited dataset size

\* Variability in audio/video quality

\* Difficulty detecting highly realistic deepfakes

\* Computational cost of multimodal models



\---



\## 🔮 Future Work



\* Improve recall (fake detection capability)

\* Use larger and diverse datasets

\* Apply advanced fusion (cross-modal attention)

\* Optimize model for real-time deployment



\---



\## 🛠️ Tech Stack



\* Python

\* PyTorch

\* OpenCV

\* Scikit-learn

\* Transformers (BERT)

\* Streamlit



\---



\## 📄 License



This project is for academic and research purposes.




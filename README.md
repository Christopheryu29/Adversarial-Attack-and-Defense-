# 🛡️ Adversarial Attack and Defense on Malware Image Classification

This repository contains implementations of **adversarial machine learning techniques** for malware classification using grayscale image representations of malware. We explore and defend against adversarial attacks such as **FGSM (Fast Gradient Sign Method)** and **data poisoning**. We also implement defense strategies like **Differential Privacy (DP)** during training.


## 📌 Features

- 🔍 **Dataset**: Grayscale malware images (512x512), grouped by malware class (20 classes total).
- 🧠 **CNN Model**: A simple convolutional neural network tailored for grayscale images.
- ⚔️ **Attacks**:
  - **FGSM**: Fast Gradient Sign Method for perturbing input images.
  - **Poisoning**: Injecting perturbed samples into the training set with incorrect labels.
- 🛡️ **Defense**:
  - **Differential Privacy**: Gradient noise injection during training for robustness.
  - **Adversarial Training**: Re-training with adversarial samples to improve model resilience.
- 📊 **Evaluation**: Accuracy before/after attacks, visualizations of predictions and perturbations.

## 🔧 Requirements

- Python 3.7+
- TensorFlow 2.x
- OpenCV
- NumPy
- scikit-learn
- Matplotlib

## 🏃 Usage

1. Defense with Differential Privacy
bash
Copy
Edit
python defense.py
This script:

Loads and preprocesses the dataset

Trains a CNN with gradient noise (DP)

Evaluates model robustness against normal test data

2. FGSM Attack & Adversarial Training
Open and run fgsm_attack.ipynb:

Generates adversarial samples using FGSM

Evaluates model performance on adversarial data

Retrains the model using adversarial training

3. Poisoning Attack
Open and run poision_attack.ipynb:

Adds noise to a subset of the training data

Assigns all poisoned samples to a target class

Trains the model and evaluates accuracy degradation

📈 Results Summary
Experiment	Clean Accuracy	Adversarial Accuracy
Baseline (Simple CNN)	82.5%	10.0% (FGSM)
Adversarial Training (FGSM)	82.5%	70.0%
Differential Privacy (DP)	~80%	~60–70% (robustness)
Poisoning Attack	77.5%	(intended misclassify rate not shown)
🧪 Note: These numbers vary depending on dataset splits and noise levels.

🖼️ Visualizations
✅ Original vs ❌ Adversarial predictions

🔬 Noise patterns from FGSM

📉 Training & validation accuracy plots

🧠 Future Work
Support for other adversarial attacks (e.g., PGD, DeepFool)

Use of pretrained models or more complex architectures

Integration with malware behavior analysis or hybrid features

📚 References
Ian J. Goodfellow et al., "Explaining and Harnessing Adversarial Examples", 2014

Nicholas Carlini et al., "Adversarial Examples Are Not Easily Detected", 2017

TensorFlow Privacy: https://github.com/tensorflow/privacy

📜 License
MIT License. Feel free to use and modify for research and education purposes.



---
title: Mini LSF Voyage
emoji: 🌍
colorFrom: blue
colorTo: indigo
app_file: app.py
pinned: false
license: cc-by-nc-4.0
short_description: Learn French Sign Language with live camera feedback 📸✋
---

**Learn French Sign Language through live camera feedback — small dataset, big journey.**

Built from scratch using custom video recordings, Mediapipe keypoints, and a Conv1D Transformer-based gesture recognition model. This app runs in your browser with real-time predictions powered by a webcam.

---

## 📖 Story of the Build

> This whole thing started as a crazy idea… and became a 4-week deep dive into sign recognition.

### 🗓 2025.05.24 — Day 1

- started project with zero dataset, just webcam and a will
- first goal: record and preprocess LSF sign clips (Bonjour, Ça va, Boire, Quoi, Au revoir)

### 📹 Week 1: Data Recording & First Failures

- edited and processed raw videos into keypoints using MediaPipe
- trained first models using LSTM and GRU with just 24 samples per class (lol)
- tried thick models with tiny dropout, nothing worked
- validation accuracy stuck around 0.11
- kept shrinking signs: 5 → 2 signs, still no good results
- lesson: more data needed — model wasn’t the problem, data was

### 📈 Week 2: Real Dataset Expansion

- spent 3 days recording new samples with different angles, clothes, lighting (even masks and hoodies)
- processed everything again
- model reached 0.58 val_acc max with bidirectional LSTMs and normalization tricks
- couldn’t break the ceiling, almost gave up
- AI said: “just get more data” — I had no more

### 🧠 Week 3: Transformers Saved Me

- first time trying Conv1D + Transformer encoder — hit 0.71 val_acc
- no overfitting, stable
- prediction code sucked at first — camera predictions failed even with strong model
- reduced to 2-class set, model hit 0.91 val_acc, loss < 0.05
- problem: double-preprocessing + missing facial keypoints

### 🔧 Week 4: Real Fixes, Real Results

- fixed input bug (removed double-preprocess)
- added full face keypoints
- chose final 5 signs that could make a short story
- final model: Conv1D + Transformer layers + residual + batchnorm + dropout
- finally got accurate live prediction working in Gradio

---

## 🧪 How It Works

- Live webcam frame → MediaPipe → landmark sequence (pose + face + hands)
- Sequence (64 frames) → Conv1D-Transformer-based model → predicted gesture
- Smart validation rules handle signs based on finger positions & face distances
- If model confidence is high and matches expected sign → advance to next scene

---

## 🎓 Tech Stack

- Python 3.10, TensorFlow / Keras
- Flask + JS + HTML
- DOCKER helpscontainer
- MediaPipe Holistic
- All signs recorded manually (me + family + a few ext videos on youtube 😂)

---

## 📦 Dataset

Small, handcrafted dataset of 5 LSF signs:

- About 40 - 60 samples per sign
- Multiple backgrounds, outfits, angles
- Converted to 1629-feature keypoint sequences

---

## 🚀 Run It

Clone the repo, install requirements, then launch:

```bash
pip install -r requirements.txt
python app.py
```

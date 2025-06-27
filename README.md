## 📖 Story of the Build

> Started as a crazy idea for my first ever project of ML … became a 4-week deep dive into LSF sign recognition.

---
> Click to watch my app demo below:

[![Watch the demo](https://img.youtube.com/vi/3Z3ZAM3X5YE/hqdefault.jpg)](https://youtu.be/zy3QDSNLnQo)

### 🗓 Week 1 (from 2024.05.26):

recorded 5 signs with webcam, processed with MediaPipe, trained LSTM/GRU — failed hard, val_acc stuck at 0.11. tried thick models, less signs… still nothing worked.

### 🗓 Week 2 (from 2024.06.02):

re-recorded with diff angles, clothes, lighting. expanded dataset to ~50 samples/sign. BiLSTM reached 0.58 val_acc — better, but not good enough. no more data, AI could not help solving the problem but "you need more data".... 😅

### 🗓 Week 3 (from 2024.06.09):

did some research, switched to Conv1D + Transformer — boom, hit 0.71 val_acc. cleaned up pipeline, added face keypoints. simplified signs to 2, hit 0.91 val_acc. fixed bugs in prediction code.

### 🗓 Week 4 (from 2024.06.16):

added smart rules (fingers, face), made it work live. POST too slow, switched to WebSocket. Flask + HTML + JS + Docker. real-time prediction in browser 💻📸

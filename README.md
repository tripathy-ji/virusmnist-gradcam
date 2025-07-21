
# 🦠 Malware Family Classification using Virus-MNIST (with Grad-CAM)

This project demonstrates a deep learning-based malware classification system trained on the **Virus-MNIST** dataset. It leverages a **Convolutional Neural Network (CNN)** to predict malware family classes from grayscale image representations of malware binaries. Additionally, it provides visual **Grad-CAM** explanations to help understand the model's predictions.

---

## 🚀 Features

- Trains a CNN on Virus-MNIST (grayscale 32x32 images)
- Visualizes predictions with Grad-CAM heatmaps
- Interactive web interface using Gradio
- Highlights Top-3 predicted malware classes with confidence scores
- Explains model decisions visually
- Ready for deployment on Hugging Face Spaces or Render

---

## 🧠 Model Architecture

- Input: 32x32 grayscale image
- 2D Convolutional Layers + MaxPooling
- Dense layers with ReLU and Dropout
- Output: Softmax classification into 8 malware families

---

## 🧾 Dataset

- **Virus-MNIST** (available on [Kaggle](https://www.kaggle.com/datasets/zihaoliu/virus-mnist))
- 32x32 grayscale image representations of malware binaries
- 8 malware classes (0 to 7)

---

## 🧪 How to Run

### 1. Train the model (already trained)
```bash
python train_model.py
```

### 2. Launch the Grad-CAM app
```bash
python app.py
```

### 3. Upload a malware image and view:
- Predicted malware family
- Top-3 class scores
- Grad-CAM explanation

---

## 📁 File Structure

```
.
├── app.py                # Gradio app with Grad-CAM
├── train_model.py        # CNN training script (optional)
├── virusmnist_cnn.h5     # Pretrained CNN model (Keras)
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## 🛠 Dependencies

Install all required packages with:

```bash
pip install -r requirements.txt
```

Main libraries used:
- TensorFlow / Keras
- NumPy, Matplotlib, OpenCV
- Gradio for UI

---

## 📦 Deployment

You can deploy this app on:
- [Hugging Face Spaces](https://huggingface.co/spaces) (Gradio interface)
- [Render](https://render.com) (Python/Flask app hosting)
- [Replit](https://replit.com) (experimental or quick demos)

---

## 📚 Acknowledgements

- Virus-MNIST Dataset by Zihao Liu (Kaggle)
- Grad-CAM Explanation technique
- TensorFlow/Keras for model training

---

## 👤 Author

Built with ❤️ as part of a deep learning + explainability portfolio project.

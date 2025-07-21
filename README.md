
# 🧬 Virus-MNIST Malware Grad-CAM Visualizer

This project is a deep learning-based web app for visualizing malware detection using **CNN and Grad-CAM**.  
It is trained on the [Virus-MNIST dataset](https://www.kaggle.com/competitions/virus-mnist-malware-image-classification/overview) and predicts malware families by interpreting 32x32 grayscale images of malicious binaries.

🛡️ It helps explain *why* the model predicted a certain malware class by highlighting the most important image regions using Grad-CAM.

---

## 🚀 Demo

**Live on Hugging Face Spaces (after deployment):**  
👉 [https://huggingface.co/spaces/your-username/virusmnist-gradcam](https://huggingface.co/spaces/your-username/virusmnist-gradcam)

### 🔍 Screenshot

<img src="demo_screenshot.png" width="800"/>

---

## 📂 Files Included

- `app.py` - Gradio app for prediction + Grad-CAM visualization
- `virusmnist_cnn.h5` - Trained CNN model file (Keras/TensorFlow)
- `requirements.txt` - List of dependencies for Hugging Face Spaces

---

## 📊 Top 3 Predictions

For each malware image you upload, the app displays:
- Top 3 predicted malware families (with probabilities)
- Grad-CAM explanation: why the model made the prediction

---

## 💡 How it Works

1. The image is resized to 32x32 and normalized.
2. A CNN classifies it into one of the 10 malware families.
3. Grad-CAM generates a heatmap showing the most activated regions.
4. The heatmap is overlaid on the original image.
5. An explanation panel shows the top 3 classes and reasoning.

---

## 🏷️ Malware Classes

- Allaple_A
- Kelihos_ver3
- Lollipop
- Obfuscator.ACY
- Ramnit
- Simda
- Tracur
- Vundo
- Kelihos_ver1
- Skintrim.N

---

## 🧪 Run Locally (Optional)

```bash
git clone https://github.com/your-username/virusmnist-gradcam.git
cd virusmnist-gradcam
pip install -r requirements.txt
python app.py
```

---

## ☁️ Deploy on Hugging Face

1. Create a new Space (Gradio)
2. Upload these files:
   - `app.py`
   - `requirements.txt`
   - `virusmnist_cnn.h5`
3. Space will auto-build and launch.

---

## 📜 License

This project is for educational use only. Dataset from [Virus-MNIST](https://www.kaggle.com/competitions/virus-mnist-malware-image-classification/overview).

---

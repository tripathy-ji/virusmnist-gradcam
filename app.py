import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2

model = load_model("virusmnist_cnn.h5")

class_labels = ['Allaple_A', 'Kelihos_ver3', 'Lollipop', 'Obfuscator.ACY', 'Ramnit',
                'Simda', 'Tracur', 'Vundo', 'Kelihos_ver1', 'Skintrim.N']

def make_gradcam_heatmap(img_array, model, last_conv_layer_name="conv2d_5"):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        output = predictions[:, class_idx]
        grads = tape.gradient(output, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        heatmap = np.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy(), class_idx.numpy(), predictions.numpy()[0]

def predict_and_explain(image):
    img = cv2.resize(np.array(image), (32, 32)) / 255.0
    input_array = np.expand_dims(img, axis=0)
    heatmap, pred_class, preds = make_gradcam_heatmap(input_array, model)
    heatmap = cv2.resize(heatmap, (32, 32))
    heatmap_img = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(np.uint8(img * 255), 0.6, heatmap_img, 0.4, 0)

    top_3 = np.argsort(preds)[-3:][::-1]
    pred_text = "**Top Predictions:**\n"
    for i in top_3:
        pred_text += f"{i+1}. {class_labels[i]} ({preds[i]*100:.2f}%)\n"
    pred_text += "\n**Why this prediction?**\nThe Grad-CAM highlights regions that most activated the CNN for the predicted class."
    return superimposed, pred_text

iface = gr.Interface(
    fn=predict_and_explain,
    inputs=gr.Image(type="pil", label="Upload a 32x32 Malware Image"),
    outputs=[gr.Image(label="Grad-CAM Visualization"), gr.Markdown(label="Explanation & Top 3 Predictions")],
    title="🧬 Virus-MNIST Malware Grad-CAM Visualizer",
    description="Upload a malware image (32x32) to visualize important regions using Grad-CAM. Model trained on Virus-MNIST."
)

iface.launch()

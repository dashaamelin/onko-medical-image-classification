import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras import Model

class GradCAM:
    """Grad-CAM (Gradient-weighted Class Activation Mapping)"""

    def __init__(self, model, layer_name=None):
        self.model = model
        self.layer_name = layer_name or self._find_last_conv_layer()
        self.grad_model = None
        self._build_grad_model()

    def _find_last_conv_layer(self):
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                print(f"Found conv layer: {layer.name}")
                return layer.name
            if hasattr(layer, 'layers'):
                for sublayer in reversed(layer.layers):
                    if isinstance(sublayer, tf.keras.layers.Conv2D):
                        print(f"Found conv layer in block: {sublayer.name}")
                        return sublayer.name
        raise ValueError("No convolutional layer found in model")

    def _build_grad_model(self):
        try:
            conv_layer = self.model.get_layer(self.layer_name)
            self.grad_model = Model(
                inputs=self.model.inputs,
                outputs=[conv_layer.output, self.model.output]
            )
        except:
            for layer in self.model.layers:
                if hasattr(layer, 'get_layer'):
                    try:
                        conv_layer = layer.get_layer(self.layer_name)
                        self.grad_model = Model(
                            inputs=self.model.inputs,
                            outputs=[conv_layer.output, self.model.output]
                        )
                        break
                    except:
                        continue
            else:
                raise ValueError(f"Layer {self.layer_name} not found")

    def generate_heatmap(self, image, class_idx=None, eps=1e-8):
        """Генерирует Grad-CAM heatmap"""
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)

        with tf.GradientTape() as tape:
            conv_output, predictions = self.grad_model(image)

            if class_idx is None:
                if self.model.output.shape[-1] == 1:
                    class_idx = 0
                else:
                    class_idx = tf.argmax(predictions[0])

            if self.model.output.shape[-1] == 1:
                loss = predictions[:, 0]
            else:
                loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_output)

        if grads is None:
            weights = tf.ones(shape=conv_output.shape[-1]) / conv_output.shape[-1]
        else:
            weights = tf.reduce_mean(grads, axis=(0, 1))

        conv_output = conv_output[0]
        heatmap = tf.reduce_sum(tf.multiply(weights, conv_output), axis=-1)

        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + eps)
        heatmap = heatmap.numpy()

        return heatmap

    def overlay_heatmap(self, image, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
        """Накладывает heatmap на изображение"""
        if len(image.shape) == 4:
            image = image[0]

        h, w = image.shape[0], image.shape[1]

        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_resized = np.uint8(255 * heatmap_resized)

        heatmap_colored = cv2.applyColorMap(heatmap_resized, colormap)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)

        return overlay

    def explain(self, image, class_idx=None, alpha=0.4, save_path=None):
        """Полный пайплайн объяснения"""
        if len(image.shape) == 3:
            image_batch = np.expand_dims(image, axis=0)
        else:
            image_batch = image
            if len(image_batch.shape) == 4:
                image = image_batch[0]

        heatmap = self.generate_heatmap(image_batch, class_idx)
        overlay = self.overlay_heatmap(image, heatmap, alpha)

        prediction = self.model.predict(image_batch, verbose=0)
        if self.model.output.shape[-1] == 1:
            prob = float(prediction[0][0])
            predicted_class = 1 if prob > 0.5 else 0
        else:
            prob = float(np.max(prediction[0]))
            predicted_class = int(np.argmax(prediction[0]))

        result = {
            'heatmap': heatmap,
            'overlay': overlay,
            'predicted_class': predicted_class,
            'confidence': prob if predicted_class == 1 else 1 - prob
        }

        if save_path:
            self.visualize(image, heatmap, overlay, predicted_class, result['confidence'], save_path)

        return result

    def visualize(self, image, heatmap, overlay, predicted_class, confidence, save_path=None):
        """Визуализирует оригинал, heatmap и наложение"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        if image.max() <= 1.0:
            img_display = (image * 255).astype(np.uint8)
        else:
            img_display = image.astype(np.uint8)

        axes[0].imshow(img_display)
        axes[0].set_title('Original Image', fontsize=12)
        axes[0].axis('off')

        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap', fontsize=12)
        axes[1].axis('off')

        axes[2].imshow(overlay)
        axes[2].set_title(f'Overlay - Prediction: Class {predicted_class} (Conf: {confidence:.3f})', fontsize=12)
        axes[2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to: {save_path}")

        plt.show()

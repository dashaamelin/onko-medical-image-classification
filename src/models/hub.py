import tensorflow as tf
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models.architectures.simple_cnn import build_simple_cnn
from src.models.architectures.transfer_models import build_efficientnet, build_resnet50

class ModelHub:
    def __init__(self, config):
        self.config = config
        self.input_shape = tuple(config['data']['image_size']) + (config['data']['num_channels'],)
        self.num_classes = config['data']['num_classes']
        self.dropout_rate = config['model'].get('dropout_rate', 0.5)
        self._model_cache = {}
        self._weights_path = "./models_saved/weights_cache"
        os.makedirs(self._weights_path, exist_ok=True)

    def get_model(self, architecture=None, load_weights_if_exists=True):
        """Возвращает модель по имени архитектуры"""
        if architecture is None:
            architecture = self.config['model']['architecture']

        if architecture in self._model_cache:
            print(f"Returning model {architecture} from cache")
            return self._model_cache[architecture]

        print(f"Building model: {architecture}")

        if architecture == 'SimpleCNN':
            model = build_simple_cnn(
                input_shape=self.input_shape,
                num_classes=self.num_classes,
                dropout_rate=self.dropout_rate
            )
            print(f"SimpleCNN: {model.count_params():,} total parameters")

        elif architecture == 'EfficientNetB0':
            pretrained = self.config['model'].get('pretrained', True)
            trainable_base = self.config['model'].get('trainable_base', False)
            fine_tune_layers = self.config['model'].get('fine_tune_layers', 0)

            model = build_efficientnet(
                input_shape=self.input_shape,
                num_classes=self.num_classes,
                pretrained=pretrained,
                trainable_base=trainable_base,
                dropout_rate=self.dropout_rate
            )

            if trainable_base and fine_tune_layers > 0:
                base_model = model.layers[0]
                if hasattr(base_model, 'layers'):
                    for layer in base_model.layers:
                        layer.trainable = False
                    for layer in base_model.layers[-fine_tune_layers:]:
                        layer.trainable = True
                    trainable_count = sum([w.shape.num_elements() for w in model.trainable_weights])
                    print(f"EfficientNetB0 fine-tuning: last {fine_tune_layers} layers unfrozen")
                    print(f"  Trainable parameters: {trainable_count:,}")

        elif architecture == 'ResNet50':
            pretrained = self.config['model'].get('pretrained', True)
            trainable_base = self.config['model'].get('trainable_base', False)
            fine_tune_layers = self.config['model'].get('fine_tune_layers', 0)

            model = build_resnet50(
                input_shape=self.input_shape,
                num_classes=self.num_classes,
                pretrained=pretrained,
                trainable_base=trainable_base,
                dropout_rate=self.dropout_rate
            )

            if trainable_base and fine_tune_layers > 0:
                base_model = model.layers[0]
                if hasattr(base_model, 'layers'):
                    for layer in base_model.layers:
                        layer.trainable = False
                    for layer in base_model.layers[-fine_tune_layers:]:
                        layer.trainable = True
                    trainable_count = sum([w.shape.num_elements() for w in model.trainable_weights])
                    print(f"ResNet50 fine-tuning: last {fine_tune_layers} layers unfrozen")
                    print(f"  Trainable parameters: {trainable_count:,}")

        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        weights_file = f"{self._weights_path}/{architecture}.weights.h5"
        if load_weights_if_exists and os.path.exists(weights_file):
            try:
                model.load_weights(weights_file)
                print(f"Loaded weights from cache: {weights_file}")
            except Exception as e:
                print(f"Failed to load weights: {e}")

        self._model_cache[architecture] = model
        return model

    def save_weights(self, architecture=None):
        """Сохраняет веса модели"""
        if architecture is None:
            architecture = self.config['model']['architecture']

        if architecture in self._model_cache:
            weights_file = f"{self._weights_path}/{architecture}.weights.h5"
            self._model_cache[architecture].save_weights(weights_file)
            print(f"Weights saved: {weights_file}")

    def list_available_models(self):
        return ['SimpleCNN', 'EfficientNetB0', 'ResNet50']

    def count_parameters(self, architecture=None):
        """Считает количество параметров модели"""
        model = self.get_model(architecture, load_weights_if_exists=False)
        total = model.count_params()
        trainable = sum([w.shape.num_elements() for w in model.trainable_weights])
        non_trainable = sum([w.shape.num_elements() for w in model.non_trainable_weights])
        return {'total': total, 'trainable': trainable, 'non_trainable': non_trainable}

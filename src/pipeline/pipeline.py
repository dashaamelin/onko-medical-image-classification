import tensorflow as tf
import os
import pickle
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.hub import ModelHub


class F1ScoreCallback(tf.keras.callbacks.Callback):
    """Callback для мониторинга F1-score на валидации"""
    def __init__(self, validation_data, threshold=0.5, patience=5):
        super().__init__()
        self.validation_data = validation_data
        self.threshold = threshold
        self.patience = patience
        self.best_f1 = 0
        self.wait = 0
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        y_true = []
        y_pred_proba = []

        for x, y in self.validation_data:
            preds = self.model.predict(x, verbose=0)
            y_true.extend(y.numpy())
            y_pred_proba.extend(preds.flatten())

        y_true = np.array(y_true)
        y_pred_proba = np.array(y_pred_proba)
        y_pred = (y_pred_proba > self.threshold).astype(int)

        f1 = f1_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)

        logs['val_f1'] = f1
        logs['val_precision'] = precision
        logs['val_recall'] = recall

        print(f" - val_f1: {f1:.4f} - val_precision: {precision:.4f} - val_recall: {recall:.4f}")

        if f1 > self.best_f1:
            self.best_f1 = f1
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
                print(f"Early stopping triggered (no F1 improvement for {self.patience} epochs)")
                if self.best_weights is not None:
                    self.model.set_weights(self.best_weights)
                    print("Restored best weights")


class MedicalImagePipeline:
    def __init__(self, config_manager, data_loader):
        self.config = config_manager.config
        self.config_manager = config_manager
        self.data_loader = data_loader
        self.model = None
        self.history = None
        self.current_epoch = 0
        self.optimal_threshold = 0.5

        self.model_hub = ModelHub(self.config)

        self.model_dir = Path("./models_saved")
        self.model_dir.mkdir(exist_ok=True)

        self.model_name = f"{self.config['project']['name']}_{self.config['model']['architecture']}"
        self.model_path = self.model_dir / f"{self.model_name}.keras"
        self.best_model_path = self.model_dir / f"{self.model_name}_best.keras"
        self.history_path = self.model_dir / f"{self.model_name}_history.pkl"

    def _build_model(self):
        model = self.model_hub.get_model()

        metrics = [
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]

        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.config['training']['learning_rate']),
            loss=self.config['model']['loss'],
            metrics=metrics
        )
        return model

    def get_callbacks(self, validation_dataset=None):
        callbacks = []

        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(self.best_model_path),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        )

        if self.config['training'].get('reduce_lr', {}).get('enabled', False):
            rl_config = self.config['training']['reduce_lr']
            callbacks.append(
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor=rl_config.get('monitor', 'val_loss'),
                    factor=rl_config.get('factor', 0.5),
                    patience=rl_config.get('patience', 5),
                    min_lr=1e-7,
                    verbose=1
                )
            )

        if validation_dataset is not None:
            callbacks.append(F1ScoreCallback(
                validation_data=validation_dataset,
                threshold=0.5,
                patience=10
            ))

        return callbacks

    def train(self):
        print("Starting training pipeline...")

        train_dataset, val_dataset = self.data_loader.load_training_data(balanced=False)

        if self.model_path.exists():
            print(f"Found saved model: {self.model_path}")
            self.model = tf.keras.models.load_model(self.model_path)
        else:
            print("Creating new model")
            self.model = self._build_model()

        callbacks = self.get_callbacks(val_dataset)
        class_weight = self.data_loader.class_weights

        print(f"Training: {self.config['training']['epochs']} epochs")

        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.config['training']['epochs'],
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1
        )

        self.history = history.history

        with open(self.history_path, 'wb') as f:
            pickle.dump(self.history, f)

        self.model.save(self.model_path)
        self.model_hub.save_weights()
        print(f"Model saved: {self.model_path}")

        return self.history

    def evaluate(self, test_dataset=None):
        if test_dataset is None:
            test_dataset = self.data_loader.load_test_data(balanced=False)

        if self.model is None:
            if self.model_path.exists():
                self.model = tf.keras.models.load_model(self.model_path)
            else:
                raise ValueError("Model not found")

        results = self.model.evaluate(test_dataset, verbose=0)

        y_true = []
        y_pred_proba = []

        for images, labels in test_dataset:
            preds = self.model.predict(images, verbose=0)
            y_true.extend(labels.numpy())
            if self.config['data']['num_classes'] == 1:
                y_pred_proba.extend(preds.flatten())
            else:
                y_pred_proba.extend(preds)

        y_true = np.array(y_true)

        metrics = {'loss': results[0], 'accuracy': results[1]}

        if self.config['data']['num_classes'] == 1:
            y_pred_proba = np.array(y_pred_proba)
            y_pred = (y_pred_proba > 0.5).astype(int)

            from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

            metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
            metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
            metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
        else:
            y_pred = np.argmax(np.array(y_pred_proba), axis=1)
            from sklearn.metrics import precision_score, recall_score, f1_score

            metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        return metrics

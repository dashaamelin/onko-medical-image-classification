import numpy as np
import tensorflow as tf
import h5py
from sklearn.utils import compute_class_weight
from pathlib import Path
from PIL import Image
import os

class MedicalDataLoader:
    def __init__(self, config):
        self.config = config
        self.class_weights = None
        self.imbalance_report = None
        self.train_images = None
        self.train_labels = None
        self.val_images = None
        self.val_labels = None
        self.test_images = None
        self.test_labels = None

    def load_training_data(self, max_samples=5000, balanced=False):
        """Загружает тренировочные данные"""
        data_format = self.config['data'].get('format', 'hdf5')

        if data_format == 'hdf5':
            all_train_images, all_train_labels = self._load_partial_data(
                self.config['data']['train_path'],
                self.config['data']['train_labels_path'],
                max_samples=max_samples,
                balanced=balanced
            )

            val_split = self.config['data'].get('val_split', 0.2)
            split_idx = int(len(all_train_images) * (1 - val_split))

            self.train_images = all_train_images[:split_idx]
            self.train_labels = all_train_labels[:split_idx]
            self.val_images = all_train_images[split_idx:]
            self.val_labels = all_train_labels[split_idx:]

        elif data_format == 'directory':
            self.train_images, self.train_labels = self._load_from_directory(
                self.config['data']['train_path']
            )
            val_split = self.config['data'].get('val_split', 0.2)
            split_idx = int(len(self.train_images) * (1 - val_split))
            
            self.val_images = self.train_images[split_idx:]
            self.val_labels = self.train_labels[split_idx:]
            self.train_images = self.train_images[:split_idx]
            self.train_labels = self.train_labels[:split_idx]

        print(f"Train/Val split: {len(self.train_images)} / {len(self.val_images)}")

        self._analyze_imbalance(self.train_labels)

        if self.config['data']['imbalance']['auto_detect']:
            self.class_weights = self._calculate_class_weights(self.train_labels)

        train_dataset = self._create_dataset(self.train_images, self.train_labels, is_training=True)
        val_dataset = self._create_dataset(self.val_images, self.val_labels, is_training=False)

        return train_dataset, val_dataset

    def load_test_data(self, max_samples=1000, balanced=False):
        """Загружает тестовые данные"""
        data_format = self.config['data'].get('format', 'hdf5')

        if data_format == 'hdf5':
            self.test_images, self.test_labels = self._load_partial_data(
                self.config['data']['test_path'],
                self.config['data']['test_labels_path'],
                max_samples=max_samples,
                balanced=balanced
            )
        elif data_format == 'directory':
            self.test_images, self.test_labels = self._load_from_directory(
                self.config['data']['test_path']
            )

        test_dataset = self._create_dataset(self.test_images, self.test_labels, is_training=False)

        return test_dataset

    def _load_partial_data(self, image_file, label_file, max_samples=5000, balanced=False):
        """Загружает часть данных из HDF5"""
        with h5py.File(image_file, 'r') as f_img, h5py.File(label_file, 'r') as f_label:
            img_key = 'x' if 'x' in f_img.keys() else list(f_img.keys())[0]
            label_key = 'y' if 'y' in f_label.keys() else list(f_label.keys())[0]

            all_labels = f_label[label_key][:].flatten()
            total_images = len(all_labels)

            selected_idx = np.arange(min(max_samples, total_images))
            selected_idx = np.sort(selected_idx)

            images = f_img[img_key][selected_idx].astype('float32')
            if images.max() > 1.0:
                images = images / 255.0

            labels = all_labels[selected_idx]

            return images, labels

    def _load_from_directory(self, path):
        """Загружает данные из директории"""
        path = Path(path)
        images = []
        labels = []
        class_names = sorted([d.name for d in path.iterdir() if d.is_dir()])
        target_size = tuple(self.config['data']['image_size'])

        print(f"   Loading from directory: {path}")
        print(f"   Found classes: {class_names}")

        for idx, class_name in enumerate(class_names):
            class_dir = path / class_name
            image_files = []
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']:
                image_files.extend(list(class_dir.glob(ext)))

            print(f"   Class {class_name} (idx={idx}): {len(image_files)} images")

            for img_path in image_files:
                try:
                    img = Image.open(img_path)
                    img = img.resize(target_size)
                    img = np.array(img)
                    if img.max() > 1.0:
                        img = img.astype('float32') / 255.0
                    if len(img.shape) == 2:
                        img = np.stack([img] * 3, axis=-1)
                    if img.shape[-1] == 4:
                        img = img[..., :3]
                    images.append(img)
                    labels.append(idx)
                except Exception as e:
                    print(f"     Warning: Could not load {img_path}: {e}")

        print(f"   Total loaded: {len(images)} images")
        return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)

    def _create_dataset(self, images, labels, is_training=False):
        """Создает tf.data.Dataset"""
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))

        if is_training:
            dataset = dataset.shuffle(buffer_size=len(images))
            if self.config['data']['augmentation']['enabled']:
                dataset = dataset.map(self._augment_image, num_parallel_calls=tf.data.AUTOTUNE)

        dataset = dataset.batch(self.config['training']['batch_size'])
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def _augment_image(self, image, label):
        """Аугментация изображения"""
        methods = self.config['data']['augmentation'].get('methods', [])

        if 'flip' in methods:
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_flip_up_down(image)
        if 'brightness' in methods:
            image = tf.image.random_brightness(image, max_delta=0.1)
        if 'contrast' in methods:
            image = tf.image.random_contrast(image, lower=0.9, upper=1.1)

        return image, label

    def _analyze_imbalance(self, labels):
        """Анализирует дисбаланс классов"""
        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)

        self.imbalance_report = {
            'total_samples': total,
            'class_distribution': dict(zip(unique, counts)),
            'percentages': {int(cls): (count/total)*100 for cls, count in zip(unique, counts)},
            'imbalance_ratio': max(counts) / min(counts) if min(counts) > 0 else float('inf')
        }

        print("DATASET BALANCE ANALYSIS")
        for cls in unique:
            print(f"Class {cls}: {counts[cls]} samples ({counts[cls]/total*100:.1f}%)")
        print(f"Imbalance ratio: {self.imbalance_report['imbalance_ratio']:.2f}")

        return self.imbalance_report

    def _calculate_class_weights(self, labels):
        """Вычисляет веса классов для борьбы с дисбалансом"""
        classes = np.unique(labels)
        weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=labels
        )
        class_weight_dict = {int(cls): float(weight) for cls, weight in zip(classes, weights)}
        print(f"Class weights: {class_weight_dict}")
        return class_weight_dict

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0, ResNet50

def build_efficientnet(input_shape, num_classes=1, pretrained=True, trainable_base=False, dropout_rate=0.5):
    """Строит EfficientNetB0 с предобучением"""
    if pretrained:
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        print("EfficientNetB0: loaded ImageNet weights")
    else:
        base_model = EfficientNetB0(
            weights=None,
            include_top=False,
            input_shape=input_shape
        )
        print("EfficientNetB0: random initialization")

    base_model.trainable = trainable_base

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)

    if num_classes == 1:
        outputs = layers.Dense(1, activation='sigmoid')(x)
    else:
        outputs = layers.Dense(num_classes, activation='softmax')(x)

    return Model(inputs=base_model.input, outputs=outputs)


def build_resnet50(input_shape, num_classes=1, pretrained=True, trainable_base=False, dropout_rate=0.5):
    """Строит ResNet50 с предобучением"""
    if pretrained:
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        print("ResNet50: loaded ImageNet weights")
    else:
        base_model = ResNet50(
            weights=None,
            include_top=False,
            input_shape=input_shape
        )
        print("ResNet50: random initialization")

    base_model.trainable = trainable_base

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)

    if num_classes == 1:
        outputs = layers.Dense(1, activation='sigmoid')(x)
    else:
        outputs = layers.Dense(num_classes, activation='softmax')(x)

    return Model(inputs=base_model.input, outputs=outputs)

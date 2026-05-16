# onko-medical-image-classification

# Анализ и сравнение методов машинного обучения для распознавания онкологии на медицинских изображениях


## Структура проекта

- `src/` — исходный код (пайплайн, модели, загрузчик данных, Grad-CAM)
- `configs/` — YAML-конфигурации для датасетов PCam, NCT-CRC-HE, ISIC

## Как запустить

1. Установите зависимости: `pip install -r requirements.txt`
2. Настройте конфиг в `configs/`
3. Запустите: `python src/main.py --config configs/pcam.yaml`

## Используемые архитектуры

- SimpleCNN (обучение с нуля)
- ResNet50
- EfficientNetB0

## Результаты

| Датасет | Лучшая модель | Accuracy |
|---------|---------------|----------|
| PCam | ResNet50 | 71.9% |
| NCT-CRC-HE | EfficientNetB0 | 92.6% |
| ISIC | ResNet50 | 56.8% |

## Автор

Сергеева Дарья Денисовна, Финансовый университет, 2026

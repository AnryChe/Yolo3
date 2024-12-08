import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from yolov3_tf2.models import YoloV3, YoloLoss
from yolov3_tf2.dataset import transform_images, transform_targets
from yolov3_tf2.utils import freeze_all

# === Конфигурация гиперпараметров ===
IMAGE_SIZE = 416  # Размер входных изображений (можно изменить)
BATCH_SIZE = 16   # Размер батча
EPOCHS = 50       # Количество эпох
LEARNING_RATE = 1e-3  # Начальная скорость обучения

# Путь к данным
DATASET_PATH = "path/to/car-object-detection-dataset"
TRAIN_ANNOTATIONS = os.path.join(DATASET_PATH, "train/_annotations.txt")
VALIDATION_ANNOTATIONS = os.path.join(DATASET_PATH, "val/_annotations.txt")
CLASSES = os.path.join(DATASET_PATH, "classes.txt")  # Список классов

# === Создание модели YOLOv3 ===
def create_model(image_size, num_classes):
    model = YoloV3(classes=num_classes)
    model.build(input_shape=(None, image_size, image_size, 3))
    return model

# === Функция подготовки данных ===
def load_dataset(annotations, classes_file, batch_size, image_size):
    dataset = tf.data.TextLineDataset(annotations)
    dataset = dataset.map(lambda x: transform_targets(x, classes_file))
    dataset = dataset.map(lambda x: transform_images(x, image_size))
    dataset = dataset.shuffle(buffer_size=512).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

# === Обучение модели ===
def train_model():
    num_classes = len(open(CLASSES).readlines())

    # Создание и компиляция модели
    model = create_model(IMAGE_SIZE, num_classes)
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss=YoloLoss())

    # Подготовка данных
    train_dataset = load_dataset(TRAIN_ANNOTATIONS, CLASSES, BATCH_SIZE, IMAGE_SIZE)
    val_dataset = load_dataset(VALIDATION_ANNOTATIONS, CLASSES, BATCH_SIZE, IMAGE_SIZE)

    # Callbacks
    checkpoint = ModelCheckpoint('yolov3_best_weights.h5', save_best_only=True, monitor='val_loss', mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min')

    # Обучение
    history = model.fit(train_dataset,
                        epochs=EPOCHS,
                        validation_data=val_dataset,
                        callbacks=[checkpoint, early_stopping])

    # Сохранение модели
    model.save('yolov3_trained.h5')

    # График обучения
    plot_training_history(history)

# === Визуализация истории обучения ===
import matplotlib.pyplot as plt

def plot_training_history(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

if __name__ == "__main__":
    train_model()
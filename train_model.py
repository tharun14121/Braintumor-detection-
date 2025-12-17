
# ===========================================================
# Brain Tumor Detection using VGG16 (Transfer Learning)
# ===========================================================

import os
import random
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import kagglehub

warnings.filterwarnings('ignore')

# -----------------------------
# Plot configuration
# -----------------------------
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 600,
    'savefig.format': 'pdf',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'axes.linewidth': 1.5,
    'lines.linewidth': 2.0
})

# ===========================================================
# STEP 1: Download Dataset
# ===========================================================
print("Downloading dataset...")
dataset_path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
print("Dataset downloaded at:", dataset_path)

train_dir = os.path.join(dataset_path, 'Training')
test_dir = os.path.join(dataset_path, 'Testing')

# ===========================================================
# STEP 2: Load and Organize Data
# ===========================================================
print("Loading image paths...")

def collect_image_paths(directory):
    paths, labels = [], []
    for label in os.listdir(directory):
        label_path = os.path.join(directory, label)
        if os.path.isdir(label_path):
            for img in os.listdir(label_path):
                paths.append(os.path.join(label_path, img))
                labels.append(label)
    return paths, labels

train_paths, train_labels = collect_image_paths(train_dir)
test_paths, test_labels = collect_image_paths(test_dir)

train_paths, train_labels = shuffle(train_paths, train_labels, random_state=42)

print(f"Training samples: {len(train_paths)} | Testing samples: {len(test_paths)}")

# ===========================================================
# STEP 3: Visualize Some Images
# ===========================================================
plt.figure(figsize=(12, 7))
plt.suptitle("Sample MRI Scans from Training Set", fontsize=18, fontweight='bold')

for i, idx in enumerate(random.sample(range(len(train_paths)), 9)):
    image = Image.open(train_paths[idx]).resize((128, 128))
    plt.subplot(3, 3, i + 1)
    plt.imshow(image)
    plt.axis('off')
    plt.title(train_labels[idx].capitalize(), fontsize=10)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# ===========================================================
# STEP 4: Class Distribution
# ===========================================================
def get_class_distribution(directory):
    return {cls: len(os.listdir(os.path.join(directory, cls)))
            for cls in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, cls))}

train_counts = get_class_distribution(train_dir)
test_counts = get_class_distribution(test_dir)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Dataset Class Distribution", fontsize=18, fontweight='bold')

axes[0].bar(train_counts.keys(), train_counts.values(), color="skyblue")
axes[0].set_title("Training Set")
axes[0].set_ylabel("Image Count")

axes[1].bar(test_counts.keys(), test_counts.values(), color="salmon")
axes[1].set_title("Testing Set")

plt.show()

# ===========================================================
# STEP 5: Data Preprocessing
# ===========================================================
IMAGE_SIZE = 224

def augment_image(image):
    """Apply simple brightness and contrast augmentations."""
    image = Image.fromarray(np.uint8(image))
    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.85, 1.15))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.85, 1.15))
    return np.array(image) / 255.0

def load_images(paths):
    images = []
    for path in paths:
        img = load_img(path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        img = img_to_array(img)
        img = augment_image(img)
        images.append(img)
    return np.array(images)

def encode_labels(labels):
    classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    return np.array([classes.index(label) for label in labels])

def data_generator(paths, labels, batch_size=12):
    """Yield batches of augmented images and encoded labels."""
    while True:
        paths, labels = shuffle(paths, labels)
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i+batch_size]
            batch_images = load_images(batch_paths)
            batch_labels = encode_labels(labels[i:i+batch_size])
            yield batch_images, batch_labels

# ===========================================================
# STEP 6: Build the Model (VGG16 + Custom Head)
# ===========================================================
print("Building the VGG16-based CNN model...")

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

# Freeze all layers except last 4
for layer in base_model.layers[:-4]:
    layer.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(len(os.listdir(train_dir)), activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ===========================================================
# STEP 7: Training the Model
# ===========================================================
batch_size = 32
train_steps = len(train_paths) // batch_size
val_steps = len(test_paths) // batch_size

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7)
]

history = model.fit(
    data_generator(train_paths, train_labels, batch_size),
    steps_per_epoch=train_steps,
    validation_data=data_generator(test_paths, test_labels, batch_size),
    validation_steps=val_steps,
    epochs=25,
    callbacks=callbacks
)

# ===========================================================
# STEP 8: Model Evaluation
# ===========================================================
print("\nEvaluating model on test data...")

test_images = load_images(test_paths)
test_labels_encoded = encode_labels(test_labels)
predictions = model.predict(test_images)
predicted_classes = np.argmax(predictions, axis=1)

class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])

print("\nClassification Report:\n")
print(classification_report(test_labels_encoded, predicted_classes, target_names=class_names))

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(test_labels_encoded, predicted_classes),
            annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# ===========================================================
# STEP 9: Predict a Single MRI Image
# ===========================================================
def detect_tumor(img_path, model, image_size=224):
    """Predict and display the tumor type for a single image."""
    img = load_img(img_path, target_size=(image_size, image_size))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    pred_idx = np.argmax(preds)
    confidence = np.max(preds)
    pred_label = class_names[pred_idx]

    result = ("No Tumor Detected" if pred_label.lower() == 'notumor'
              else f"Tumor Detected ({pred_label.capitalize()})")

    plt.imshow(load_img(img_path))
    plt.axis('off')
    plt.title(f"{result}\nConfidence: {confidence*100:.2f}%", fontsize=12)
    plt.show()

# Example image test
sample_image = os.path.join(test_dir, "glioma", "Te-gl_0028.jpg")
print("\nTesting on one MRI image...")
detect_tumor(sample_image, model)

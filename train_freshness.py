import tensorflow as tf
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split


# =====================================================
# SETTINGS
# =====================================================

DATASET_DIR = Path("dataset/freshness")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 123
EPOCHS = 10


# =====================================================
# FIND CLASSES
# =====================================================

class_names = sorted([
    folder.name
    for folder in DATASET_DIR.iterdir()
    if folder.is_dir()
])

print("\nClasses found:")
print(class_names)

if class_names != ["fresh", "spoiled"]:
    raise ValueError(
        "Dataset must contain exactly two folders: fresh and spoiled"
    )


# =====================================================
# COLLECT IMAGE PATHS
# =====================================================

image_paths = []
labels = []

valid_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
}

for label, class_name in enumerate(class_names):

    class_folder = DATASET_DIR / class_name

    files = [
        file
        for file in class_folder.rglob("*")
        if file.suffix.lower() in valid_extensions
    ]

    print(
        f"{class_name} images: {len(files)}"
    )

    for file in files:
        image_paths.append(str(file))
        labels.append(label)


image_paths = np.array(image_paths)
labels = np.array(labels)


# =====================================================
# TOTAL IMAGES
# =====================================================

print("\nTotal images:", len(image_paths))


# =====================================================
# TRAIN / VALIDATION SPLIT
# =====================================================

train_paths, val_paths, train_labels, val_labels = train_test_split(
    image_paths,
    labels,
    test_size=0.20,
    random_state=SEED,
    stratify=labels
)


print("\nTraining images:", len(train_paths))
print("Validation images:", len(val_paths))


# =====================================================
# VALIDATION CLASS COUNTS
# =====================================================

print("\nValidation class counts:")

for i, name in enumerate(class_names):

    count = np.sum(val_labels == i)

    print(
        f"{name}: {count}"
    )


# =====================================================
# LOAD IMAGE
# =====================================================

def load_image(path, label):

    image = tf.io.read_file(path)

    image = tf.io.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape(
        [None, None, 3]
    )

    image = tf.image.resize(
        image,
        IMG_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    )

    return image, label


# =====================================================
# CREATE TRAIN DATASET
# =====================================================

train_ds = tf.data.Dataset.from_tensor_slices(
    (train_paths, train_labels)
)

train_ds = train_ds.shuffle(
    buffer_size=len(train_paths),
    seed=SEED
)

train_ds = train_ds.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

train_ds = train_ds.batch(
    BATCH_SIZE
)

train_ds = train_ds.prefetch(
    tf.data.AUTOTUNE
)


# =====================================================
# CREATE VALIDATION DATASET
# =====================================================

val_ds = tf.data.Dataset.from_tensor_slices(
    (val_paths, val_labels)
)

val_ds = val_ds.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

val_ds = val_ds.batch(
    BATCH_SIZE
)

val_ds = val_ds.prefetch(
    tf.data.AUTOTUNE
)


# =====================================================
# DATA AUGMENTATION
# =====================================================

data_augmentation = tf.keras.Sequential([

    tf.keras.layers.RandomFlip(
        "horizontal"
    ),

    tf.keras.layers.RandomRotation(
        0.10
    ),

    tf.keras.layers.RandomZoom(
        0.15
    ),

    tf.keras.layers.RandomContrast(
        0.15
    )

])


# =====================================================
# MOBILE NET V2
# =====================================================

print("\nLoading MobileNetV2...")

base_model = tf.keras.applications.MobileNetV2(

    input_shape=(
        224,
        224,
        3
    ),

    include_top=False,

    weights="imagenet"
)

base_model.trainable = False


# =====================================================
# CREATE MODEL
# =====================================================

inputs = tf.keras.Input(
    shape=(
        224,
        224,
        3
    )
)


# Data augmentation

x = data_augmentation(
    inputs
)


# MobileNetV2 preprocessing

x = tf.keras.applications.mobilenet_v2.preprocess_input(
    x
)


# MobileNetV2

x = base_model(
    x,
    training=False
)


# Global Average Pooling

x = tf.keras.layers.GlobalAveragePooling2D()(x)


# Dropout

x = tf.keras.layers.Dropout(
    0.4
)(x)


# Output layer

outputs = tf.keras.layers.Dense(
    2,
    activation="softmax"
)(x)


# Create model

model = tf.keras.Model(
    inputs,
    outputs
)


# =====================================================
# COMPILE MODEL
# =====================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


# =====================================================
# MODEL SUMMARY
# =====================================================

print("\nModel created successfully.")

model.summary()


# =====================================================
# SAVE CLASS NAMES
# =====================================================

with open(
    "model/freshness_class_names.json",
    "w"
) as f:

    json.dump(
        class_names,
        f
    )


print(
    "\nClass names saved:"
)

print(
    class_names
)


# =====================================================
# START TRAINING
# =====================================================

print("\n====================================")
print("   STARTING FRESHNESS MODEL TRAINING")
print("====================================")


history = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=EPOCHS
)


# =====================================================
# TRAINING COMPLETED
# =====================================================

print("\n====================================")
print("       TRAINING COMPLETED")
print("====================================")


# =====================================================
# SAVE MODEL
# =====================================================

model.save(
    "model/freshness_model.keras"
)


print(
    "\nFreshness model saved successfully!"
)

print(
    "Location: model/freshness_model.keras"
)

print(
    "Class names saved successfully!"
)

print(
    "Location: model/freshness_class_names.json"
)

print(
    "\nClasses:",
    class_names
)
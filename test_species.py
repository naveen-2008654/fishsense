import tensorflow as tf
import json
from pathlib import Path

# Load species model
model = tf.keras.models.load_model("model/species_model.keras")

# Load species names
with open("model/species_class_names.json", "r") as f:
    class_names = json.load(f)

print("Classes:")
print(class_names)

# Take the first image from the first species folder
species_folders = [
    folder for folder in Path("dataset/species").iterdir()
    if folder.is_dir()
]

species_folder = species_folders[0]

images = [
    file for file in species_folder.rglob("*")
    if file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
]

image_path = images[0]

print("\nTesting image:", image_path)

# Load image
image = tf.keras.utils.load_img(
    image_path,
    target_size=(224, 224)
)

image = tf.keras.utils.img_to_array(image)
image = tf.expand_dims(image, axis=0)

# Predict
prediction = model.predict(image, verbose=0)[0]

print("\nTop prediction:")

predicted_index = prediction.argmax()
predicted_class = class_names[predicted_index]

print("Species:", predicted_class)
print("Confidence:", round(float(prediction[predicted_index]) * 100, 2), "%")
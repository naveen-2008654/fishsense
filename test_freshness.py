import tensorflow as tf
import json
from pathlib import Path

# Load model
model = tf.keras.models.load_model("model/freshness_model.keras")

# Load class names
with open("model/freshness_class_names.json", "r") as f:
    class_names = json.load(f)

print("Classes:", class_names)

# Take one image from fresh folder
spoiled_images = list(Path("dataset/freshness/spoiled").glob("*"))
image_path = spoiled_images[0]

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

print("\nPrediction probabilities:")
for name, probability in zip(class_names, prediction):
    print(name, ":", round(float(probability) * 100, 2), "%")

predicted_class = class_names[prediction.argmax()]

print("\nFINAL PREDICTION:", predicted_class)
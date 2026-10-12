from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import json
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ==============================
# LOAD SPECIES MODEL
# ==============================

species_model = load_model("model/species_model.keras")

with open("model/species_class_names.json", "r") as f:
    species_class_names = json.load(f)


# ==============================
# LOAD FRESHNESS MODEL
# ==============================

freshness_model = load_model("model/freshness_model.keras")

with open("model/freshness_class_names.json", "r") as f:
    freshness_class_names = json.load(f)


# ==============================
# PREDICTION FUNCTION
# ==============================

def predict_image(image_path, model, class_names):

    image = Image.open(image_path).convert("RGB")

    # Resize exactly like training
    image = image.resize((224, 224))

    # Convert image to numpy
    image_array = np.array(image, dtype=np.float32)

    # IMPORTANT:
    # Do NOT divide by 255 here.
    # The trained models perform their own preprocessing.

    image_array = np.expand_dims(image_array, axis=0)

    # Prediction
    predictions = model.predict(image_array, verbose=0)[0]

    predicted_index = int(np.argmax(predictions))

    predicted_class = class_names[predicted_index]

    confidence = float(predictions[predicted_index]) * 100

    return predicted_class, confidence


# ==============================
# HOME PAGE
# ==============================

@app.route("/", methods=["GET", "POST"])
def home():

    species_result = None
    freshness_result = None

    species_confidence = None
    freshness_confidence = None

    image_path = None
    error = None

    if request.method == "POST":

        # Check image
        if "fish_image" not in request.files:

            error = "Please select a fish image."

            return render_template(
                "index.html",
                error=error
            )

        file = request.files["fish_image"]

        # Check filename
        if file.filename == "":

            error = "Please select a fish image."

            return render_template(
                "index.html",
                error=error
            )

        # Save uploaded image
        file_path = os.path.join(
            UPLOAD_FOLDER,
            file.filename
        )

        file.save(file_path)

        # ==============================
        # SPECIES PREDICTION
        # ==============================

        species_result, species_confidence = predict_image(
            file_path,
            species_model,
            species_class_names
        )

        # ==============================
        # FRESHNESS PREDICTION
        # ==============================

        freshness_result, freshness_confidence = predict_image(
            file_path,
            freshness_model,
            freshness_class_names
        )

        # Image path for webpage
        image_path = "/" + file_path.replace("\\", "/")

    return render_template(
        "index.html",
        species_result=species_result,
        freshness_result=freshness_result,
        species_confidence=species_confidence,
        freshness_confidence=freshness_confidence,
        image_path=image_path,
        error=error
    )


# ==============================
# RUN FLASK
# ==============================

if __name__ == "__main__":
    app.run(debug=True)
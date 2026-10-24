import os
import gc
import json
import uuid
from pathlib import Path

# Use CPU instead of trying to initialize a GPU on Render.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from PIL import Image
import numpy as np


# =====================================
# 1. INITIALIZE FLASK
# =====================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


# =====================================
# 2. MODEL PATHS
# =====================================

SPECIES_MODEL_PATH = MODEL_DIR / "species_model.keras"
SPECIES_CLASSES_PATH = MODEL_DIR / "species_class_names.json"

FRESHNESS_MODEL_PATH = MODEL_DIR / "freshness_model.keras"
FRESHNESS_CLASSES_PATH = MODEL_DIR / "freshness_class_names.json"


# =====================================
# 3. CHECK FILES
# =====================================

required_files = [
    SPECIES_MODEL_PATH,
    SPECIES_CLASSES_PATH,
    FRESHNESS_MODEL_PATH,
    FRESHNESS_CLASSES_PATH
]

for path in required_files:
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")


# =====================================
# 4. LOAD CLASS NAMES
# =====================================

def load_class_names(path):

    with open(path, "r", encoding="utf-8") as file:
        names = json.load(file)

    if isinstance(names, list):
        return names

    if isinstance(names, dict):
        return [
            names[str(i)] if str(i) in names else names[i]
            for i in range(len(names))
        ]

    raise ValueError("Invalid class names file")


species_class_names = load_class_names(
    SPECIES_CLASSES_PATH
)

freshness_class_names = load_class_names(
    FRESHNESS_CLASSES_PATH
)


# =====================================
# 5. PREDICTION FUNCTION
# =====================================

def predict_image(image_path, model_path, class_names):

    model = None

    try:

        # Load only one model at a time.
        K.clear_session()
        gc.collect()

        print("Loading model:", model_path.name)

        model = load_model(
            str(model_path),
            compile=False
        )

        input_shape = model.input_shape

        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        height = input_shape[1] or 224
        width = input_shape[2] or 224
        channels = input_shape[3] if len(input_shape) == 4 else 3

        if channels != 3:
            raise ValueError("Model must accept RGB images")

        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img = img.resize((width, height))
            image_array = np.asarray(
                img, dtype=np.float32
            )

        image_array = np.expand_dims(
            image_array, axis=0
        )

        # Match preprocessing used during training.
        # If training divided images by 255, apply
        # the same preprocessing here.

        predictions = model.predict(
            image_array,
            verbose=0
        )

        if isinstance(predictions, list):
            predictions = predictions[0]

        predictions = np.asarray(predictions)[0]

        if predictions.ndim != 1:
            raise ValueError("Unexpected model output")

        predicted_index = int(np.argmax(predictions))

        if predicted_index >= len(class_names):
            raise ValueError("Class names do not match model")

        predicted_class = class_names[predicted_index]

        confidence = float(
            predictions[predicted_index]
        ) * 100

        return predicted_class, confidence

    finally:

        # Release model memory after prediction.
        if model is not None:
            del model

        K.clear_session()
        gc.collect()


# =====================================
# 6. HOME PAGE
# =====================================

@app.route("/", methods=["GET", "POST"])
def home():

    context = {
        "species_result": None,
        "freshness_result": None,
        "species_confidence": None,
        "freshness_confidence": None,
        "image_path": None,
        "error": None
    }

    if request.method == "POST":

        uploaded_file = request.files.get("fish_image")

        if uploaded_file is None or not uploaded_file.filename:
            context["error"] = "Please select a fish image."
            return render_template("index.html", **context)

        extension = Path(uploaded_file.filename).suffix.lower()

        if extension not in {".jpg", ".jpeg", ".png", ".webp"}:
            context["error"] = "Please upload a valid fish image."
            return render_template("index.html", **context)

        filename = f"{uuid.uuid4().hex}{extension}"
        file_path = UPLOAD_FOLDER / filename

        try:

            uploaded_file.save(str(file_path))

            # First: Species prediction.
            species_result, species_confidence = predict_image(
                file_path,
                SPECIES_MODEL_PATH,
                species_class_names
            )

            # Second: Freshness prediction.
            freshness_result, freshness_confidence = predict_image(
                file_path,
                FRESHNESS_MODEL_PATH,
                freshness_class_names
            )

            context.update({
                "species_result": species_result,
                "freshness_result": freshness_result,
                "species_confidence": round(species_confidence, 2),
                "freshness_confidence": round(freshness_confidence, 2),
                "image_path": f"/static/uploads/{filename}"
            })

        except Exception:

            app.logger.exception("FishSense prediction error")

            context["error"] = (
                "Unable to analyze fish image. "
                "Please check Render application logs."
            )

    return render_template("index.html", **context)


# =====================================
# 7. IMAGE SIZE ERROR
# =====================================

@app.errorhandler(413)
def file_too_large(error):

    return render_template(
        "index.html",
        species_result=None,
        freshness_result=None,
        species_confidence=None,
        freshness_confidence=None,
        image_path=None,
        error="Image must be under 10 MB."
    ), 413


# =====================================
# 8. RUN APP
# =====================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )
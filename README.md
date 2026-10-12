# Fishsense

Fish species and freshness detection web app built with Flask and TensorFlow.

## Features
- Identifies the fish species from an uploaded image
- Predicts whether the fish is fresh or spoiled

## How to run
pip install -r requirements.txt
python app.py

Then open http://127.0.0.1:5000 in your browser.

## Files
- app.py: Flask web app
- train_species.py, train_freshness.py: model training
- test_species.py, test_freshness.py, validate_model.py: testing
- model/: trained models (stored with Git LFS)

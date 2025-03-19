import pytest
import numpy as np
import pandas as pd
import cloudpickle  # <-- Replacing joblib with cloudpickle
import os
import sys


# Add the project directory to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Explicitly import the custom model class
from Model.Model import MyLogisticRegression

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_test.pkl")
PREPROCESS_PATH = os.path.join(BASE_DIR, "preprocess_test.pkl")
BRAND_MEANS_PATH = os.path.join(BASE_DIR, "brand_means.pkl")

print("Preprocessor path:", PREPROCESS_PATH)
print("File size (bytes):", os.path.getsize(PREPROCESS_PATH))

# --- Load with cloudpickle instead of joblib ---
with open(PREPROCESS_PATH, "rb") as f:
    preprocessor = cloudpickle.load(f)

with open(BRAND_MEANS_PATH, "rb") as f:
    brand_means = cloudpickle.load(f)

# Initialize your custom model (if you rely on an internal .predict logic)
with open(MODEL_PATH, "rb") as f:
    model = cloudpickle.load(f)

# Define sample input as DataFrame
columns = ["year", "km_driven", "seller_type", "transmission", "engine", "max_power", "brand"]
sample_input = pd.DataFrame([[2015, 50000, "Individual", "Automatic", 1000, 80, "Audi"]], columns=columns)

# Map brand to encoded value
sample_input["brand_encoded"] = sample_input["brand"].map(brand_means)
sample_input["brand_encoded"] = sample_input["brand_encoded"].fillna(np.mean(list(brand_means.values())))
sample_input = sample_input.drop(columns=["brand"])  # Drop original brand column

def test_model_input():
    """Test if the model takes expected input format."""
    try:
        preprocess_input = preprocessor.transform(sample_input)
        model.predict(preprocess_input, is_test=True)
    except Exception as e:
        pytest.fail(f"Model failed to take expected input: {e}")

def test_model_output():
    """Test if model output shape is correct."""
    preprocess_input = preprocessor.transform(sample_input)
    _, prediction = model.predict(preprocess_input, is_test=True)
    assert prediction.shape == (1,), "Output shape is incorrect"



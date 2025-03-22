import pytest
import numpy as np
import pandas as pd
import cloudpickle  # <-- Replacing joblib with cloudpickle
import os
import sys
from Utils.utils import load_latest_model


# Add the project directory to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

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
#
with open(BRAND_MEANS_PATH, "rb") as f:
    brand_means = cloudpickle.load(f)
#
# # Initialize your custom model (if you rely on an internal .predict logic)
# with open(MODEL_PATH, "rb") as f:
#     model = cloudpickle.load(f)

# Define sample input as DataFrame
columns = ["year", "km_driven", "seller_type", "transmission", "engine", "max_power", "brand"]
sample_input = pd.DataFrame([[2015, 50000, "Individual", "Automatic", 1000, 80, "Audi"]], columns=columns)

# Map brand to encoded value
sample_input["brand_encoded"] = sample_input["brand"].map(brand_means)
sample_input["brand_encoded"] = sample_input["brand_encoded"].fillna(np.mean(list(brand_means.values())))
sample_input = sample_input.drop(columns=["brand"])  # Drop original brand column

def test_load_model():
    model = load_latest_model()
    assert model

def test_model_input():
    """Test if the model takes expected input format."""
    model2 = load_latest_model()
    try:
        preprocess_input = preprocessor.transform(sample_input)

        # Ensure correct number of features
        num_features = preprocess_input.shape[1]  # Get feature count
        expected_features = ["bias"] + [f"feature_{i}" for i in range(num_features)]

        # Ensure bias term is correctly added
        preprocess_input = np.insert(preprocess_input, 0, 1, axis=1)  # Add bias column
        preprocess_input = pd.DataFrame(preprocess_input, columns=expected_features)

        print("Preprocessed input shape:", preprocess_input.shape)  # Debugging
        print("Expected model signature:", model2.metadata.signature.inputs)  # Debugging

        # Pass to model
        model2.predict(preprocess_input)
    except Exception as e:
        pytest.fail(f"Model failed to take expected input: {e}")

def test_model_output():
    """Test if model output shape is correct."""
    model2 = load_latest_model()
    wrapper_model = model2._model_impl.python_model
    actual_model = wrapper_model.model
    if hasattr(actual_model, "_coeff_and_biases"):
        # Ensure weights exist before accessing shape
        if hasattr(actual_model, "W"):
            num_features = actual_model.W.shape[0] - 1  # Exclude bias term
            feature_names = ["feature_" + str(i) for i in range(num_features)]

            # Get coefficients and bias
            coef_df, bias = actual_model._coeff_and_biases(feature_names)

            assert coef_df.shape == (10, 4) and bias.shape == (4,), \
        f"Output shape is incorrect. Got coef shape {coef_df.shape}, bias shape {bias.shape}"

    print("Model output shape test passed!")


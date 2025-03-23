import cloudpickle
import os
import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

# Load environment variables from .env (useful for local testing)
load_dotenv()

# Define Model Name & Tracking URI
# Ensure environment variables are loaded (these come from GitHub Secrets)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")
model_name = os.getenv("APP_MODEL_NAME")

if not MLFLOW_TRACKING_URI or not model_name:
    raise ValueError("MLFLOW_TRACKING_URI or APP_MODEL_NAME is not set. Ensure they are defined in GitHub Secrets.")

# Set MLflow Tracking URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Initialize MlflowClient (MUST be done inside the script)
client = MlflowClient()

# Define Cache Directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_CACHE_DIR = os.path.join(PROJECT_ROOT, "models")

def save(filename:str, obj:object):
    """Save object using cloudpickle"""
    with open(filename, 'wb') as handle:
        cloudpickle.dump(obj, handle)

def load(filename:str) -> object:
    """Load object using cloudpickle"""
    with open(filename, 'rb') as handle:
        return cloudpickle.load(handle)

def load_latest_model(stage="Staging"):
    """Load the latest MLflow model from the specified stage"""

    if not model_name:
        raise ValueError("APP_MODEL_NAME is not set. Ensure it is defined in GitHub Secrets.")

    cache_path = os.path.join(MODEL_CACHE_DIR, stage)
    os.makedirs(cache_path, exist_ok=True)  # Ensure directory exists

    try:
        # Fetch latest model version from MLflow
        versions = client.get_latest_versions(name=model_name, stages=[stage])
        if not versions:
            print(f"No model found in '{stage}' stage.")
            return None

        latest_version = versions[0].version
        print(f"Found model '{model_name}', Version: {latest_version}, Stage: {stage}")

        # **GitHub CI/CD always fetches the model from MLflow**
        model_uri = f"models:/{model_name}/{latest_version}"
        model = mlflow.pyfunc.load_model(model_uri=model_uri)

        # **Optional: Cache the model locally for debugging**
        local_model_path = os.path.join(cache_path, f"{model_name}.pkl")
        mlflow.pyfunc.save_model(model, local_model_path)
        print(f"Model saved locally at {local_model_path}")

        return model

    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def register_model_to_production():
    """Promote Staging model to Production in MLflow"""
    try:
        versions = client.get_latest_versions(name=model_name, stages=["Staging"])
        if not versions:
            print(f"No model found in Staging for {model_name}.")
            return

        version = versions[0].version
        print(f"Promoting Model {model_name} version {version} to Production...")

        client.transition_model_version_stage(
            name=model_name, version=version, stage="Production", archive_existing_versions=True
        )

        print(f"Model {model_name} version {version} is now in Production!")

    except Exception as e:
        print(f"Error in transitioning model to Production: {e}")


import cloudpickle
import os
import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

load_dotenv()

def save(filename:str, obj:object):
    with open(filename, 'wb') as handle:
        cloudpickle.dump(obj, handle,)

def load(filename:str) -> object:
    with open(filename, 'rb') as handle:
        b = cloudpickle.load(handle)
    return b

# Set MLflow Tracking URI
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

# Set Authentication (if required)
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

# Define Model Name
model_name = os.getenv("APP_MODEL_NAME")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_CACHE_DIR = os.path.join(PROJECT_ROOT, "models")
client = MlflowClient()
def load_latest_model(stage="Staging"):

    client = MlflowClient()

    # Create cache directory if it doesn't exist
    cache_path = os.path.join(MODEL_CACHE_DIR, stage)
    os.makedirs(cache_path, exist_ok=True)

    try:
        # Fetch latest model version in the given stage
        versions = client.get_latest_versions(name=model_name, stages=[stage])
        if not versions:
            print(f"No model found in '{stage}' stage.")
            return None

        latest_version = versions[0].version  # Get latest version
        print(f"Found model '{model_name}', Version: {latest_version}, Stage: {stage}")

        # Define local path to save the model
        local_model_path = os.path.join(cache_path, f"{model_name}")

        # Check if model already exists in cache
        if os.path.exists(local_model_path):
            print(f"Loading model from cache: {local_model_path}")
            return load(local_model_path)
        else:
            model_uri = f"models:/{model_name}/{latest_version}"
            model = mlflow.pyfunc.load_model(model_uri=model_uri)
            save(local_model_path, model)
            print(f"Model saved locally at {local_model_path}")

            return model

    except Exception as e:
        print(f" Error loading model: {e}")
        return None


def register_model_to_production():
    """Promote Staging model to Production in MLflow"""
    client = MlflowClient()

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

load_latest_model()
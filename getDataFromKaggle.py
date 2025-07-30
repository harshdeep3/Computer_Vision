
import os
import kagglehub
from kaggle.api.kaggle_api_extended import KaggleApi
import json


def load_kaggle_json(kaggle_json_path: str):
    """
    Load Kaggle API credentials from kaggle.json file.
    """
    with open(kaggle_json_path) as f:
        creds = json.load(f)
    
    # Set environment variable for API key manually
    os.environ['KAGGLE_USERNAME'] = creds['username']
    os.environ['KAGGLE_KEY'] = creds['key']
    
    print("Kaggle credentials loaded successfully.", creds)


def get_kaggle_api(kaggle_json_path: str= None) -> KaggleApi:
    """
    Get the Kaggle API client.
    """
    if kaggle_json_path is not None:
        load_kaggle_json(kaggle_json_path)
    
    api = KaggleApi()
    api.authenticate()
    return api


def get_data_from_kaggle(dataset_name: str, save_path: str, kaggle_json_path: str=None):

    api = get_kaggle_api(kaggle_json_path)
    # Download latest version
    path = api.dataset_download_files(dataset_name, path=save_path, unzip=True)

    print(f"Dataset {dataset_name} downloaded to {path}")


if __name__ == "__main__":
    dataset_name = "meetnagadia/human-action-recognition-har-dataset"
    save_path = "data/human_action_recognition"

    get_data_from_kaggle(dataset_name, save_path)

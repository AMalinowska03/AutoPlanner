import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

download_path = "./kaggle_datasets"

dataset_name = "zara2099/human-resource-allocation-dataset"
api.dataset_download_files(dataset_name, path=download_path, unzip=True)

dataset_name = "ak99994/tasks-data-optimization-purpose"
api.dataset_download_files(dataset_name, path=download_path, unzip=True)

dataset_name = "ustunahmet/cognitive-performance-and-biometric-optimization-ds"
api.dataset_download_files(dataset_name, path=download_path, unzip=True)

print("Files downloaded and unpacked")
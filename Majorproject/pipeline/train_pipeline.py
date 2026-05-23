import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from Majorproject.components.data_ingestion import DataIngestion
from Majorproject.components.data_transformation import DataTransformation
from Majorproject.components.model_trainer import ModelTrainer
from Majorproject.exception import CustomException
from Majorproject.logger import logging


def main():
    try:
        ingestion = DataIngestion()
        train_data_path, test_data_path, _ = ingestion.initiate_data_ingestion()

        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
            train_data_path, test_data_path
        )

        model_trainer = ModelTrainer()
        r2_score = model_trainer.initiate_model_trainer(train_arr, test_arr)
        print(f"Training complete. R2 score: {r2_score}")

    except Exception as e:
        logging.error(str(e))
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()

import os
import sys
import pickle
import dill

from Majorproject.exception import CustomException


def save_object(file_path: str, obj: object) -> None:
    try:
        # Extract directory path
        dir_path = os.path.dirname(file_path)

        # Create directory if it does not exist
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        # Open file in write-binary mode
        with open(file_path, "wb") as file_obj:

            # Save object using dill
            dill.dump(obj, file_obj)

    except Exception as e:

        # Raise custom exception
        raise CustomException(e, sys)
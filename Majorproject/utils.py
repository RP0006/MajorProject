import os
import sys
import pickle
import dill
from sklearn.metrics import r2_score
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
    
def evaluate_models(X_train, y_train, X_test, y_test, models: dict) -> dict:
    try:
        report = {}

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)
            train_model_score = r2_score(y_train, model.predict(X_train))
            test_model_score = r2_score(y_test, y_test_pred)
            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
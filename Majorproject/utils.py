import os
import sys
import dill

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from Majorproject.exception import CustomException


def save_object(file_path: str, obj: object) -> None:
    try:
        # Get directory path
        dir_path = os.path.dirname(file_path)

        # Create directory if not exists
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        # Save object
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:

        report = {}

        for i in range(len(list(models))):

            model_name = list(models.keys())[i]
            model = list(models.values())[i]

            para = params[model_name]

            print(f"\nRunning GridSearchCV for {model_name}")

            gs = GridSearchCV(
                estimator=model,
                param_grid=para,
                cv=3,
                verbose=1,
                n_jobs=-1
            )

            # Train GridSearchCV
            gs.fit(X_train, y_train)

            # Set best parameters
            model.set_params(**gs.best_params_)

            # Train model with best params
            model.fit(X_train, y_train)

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Scores
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Store test score
            report[model_name] = test_model_score

            print(f"\n{model_name} Results")
            print(f"Best Parameters: {gs.best_params_}")
            print(f"Train R2 Score: {train_model_score}")
            print(f"Test R2 Score: {test_model_score}")

        return report

    except Exception as e:
        raise CustomException(e, sys)
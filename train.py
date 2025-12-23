import logging
import os

import joblib
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge

from utils import load_data, split_data

# Configure logging
logging.basicConfig(
    filename="train.log",
    level=logging.INFO,
    format="%(asctime)s %(filename)s:%(lineno)d %(levelname)s %(message)s",
)


def log_print(msg):
    # Print to console and log to file
    print(msg)
    logging.info(msg)


def train_model(model, name, X_train, y_train, X_columns):
    log_print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

    # Make data directory if not exists
    os.makedirs("data", exist_ok=True)

    # Save model
    model_filename = f"{name.replace(' ', '_').lower()}.joblib"
    model_path = os.path.join("data", model_filename)
    joblib.dump(model, model_path)
    log_print(f"Model saved to {model_path}")

    # Print coefficients
    log_print(f"{name} Coefficients:")
    coef_dict = dict(zip(X_columns, model.coef_))
    for feat, coef in coef_dict.items():
        log_print(f"{feat}: {coef:.4f}")
    log_print(f"intercept_: {model.intercept_:.4f}")

    return model


def train_all():
    try:
        log_print("Starting Training Pipeline...")

        log_print("Loading data...")
        X, y = load_data()
        log_print(f"Data loaded. Shape: {X.shape}")

        X_train, X_test, y_train, y_test = split_data(X, y)

        # Function 1: Linear Regression
        train_model(
            LinearRegression(),
            "Linear Regression",
            X_train,
            y_train,
            X.columns,
        )

        # Function 2: Ridge Regression
        alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
        log_print("\n--- Function 2: Ridge Regression ---")
        for alpha in alphas:
            train_model(
                Ridge(alpha=alpha),
                f"Ridge alpha={alpha}",
                X_train,
                y_train,
                X.columns,
            )

        # Function 3: Lasso Regression
        log_print("\n--- Function 3: Lasso Regression ---")
        for alpha in alphas:
            train_model(
                Lasso(alpha=alpha),
                f"Lasso alpha={alpha}",
                X_train,
                y_train,
                X.columns,
            )

        # Function 4: ElasticNet Regression
        log_print("\n--- Function 4: ElasticNet Regression ---")
        for alpha in alphas:
            train_model(
                ElasticNet(alpha=alpha, l1_ratio=0.5),
                f"ElasticNet alpha={alpha}",
                X_train,
                y_train,
                X.columns,
            )

        log_print("\nAll training completed.")

    except Exception as e:
        logging.error(f"Error during training: {str(e)}", exc_info=True)
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    train_all()

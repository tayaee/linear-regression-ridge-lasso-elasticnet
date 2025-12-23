import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


def load_data(filepath="car-mpg.csv"):
    """
    Load car-mpg dataset and preprocess it.
    """
    df = pd.read_csv(filepath)

    # Check for missing values '?' and replace with NaN
    df.replace("?", pd.NA, inplace=True)

    # Convert columns to numeric, coercing errors
    cols_to_numeric = ["hp"]
    for col in cols_to_numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with missing target 'mpg' if any (though usually we impute features)
    df.dropna(subset=["mpg"], inplace=True)

    X = df.drop("mpg", axis=1)
    y = df["mpg"]

    # Handle missing values in features
    imputer = SimpleImputer(strategy="mean")
    # We only have one potential object col converted to numeric (horsepower).
    # Others are likely numeric. Origin and model year are categorical-like but treated as numeric often in simple regression.
    # car_name is string, we should probably drop it for regression or encode it.
    # Usually car_name is high cardinality, dropping for this exercise relative to requirement.
    if "car_name" in X.columns:
        X = X.drop("car_name", axis=1)

    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    return X_imputed, y


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

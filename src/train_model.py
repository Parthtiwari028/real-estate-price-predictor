import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import numpy as np


# Load final dataset
df = pd.read_csv("data/final_housing.csv")

# -----------------------------------
# Select features and target
# -----------------------------------

X = df[
    [
        "area_type",
        "location",
        "total_sqft",
        "bath",
        "balcony",
        "bhk",
        "total_rooms"
    ]
]

y = df["price"]


# -----------------------------------
# Categorical and numerical columns
# -----------------------------------

categorical_features = [
    "area_type",
    "location"
]

numerical_features = [
    "total_sqft",
    "bath",
    "balcony",
    "bhk",
    "total_rooms"
]


# -----------------------------------
# Preprocessing
# -----------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        ),
        (
            "numerical",
            SimpleImputer(strategy="median"),
            numerical_features
        )
    ]
)


# -----------------------------------
# Linear Regression model
# -----------------------------------

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ]
)


# -----------------------------------
# Train-Test Split
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


# -----------------------------------
# Train model
# -----------------------------------

model.fit(X_train, y_train)


# -----------------------------------
# Prediction
# -----------------------------------

y_pred = model.predict(X_test)


# -----------------------------------
# Evaluation
# -----------------------------------

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)


print("\n--- LINEAR REGRESSION RESULTS ---")

print("MAE:", round(mae, 2))

print("RMSE:", round(rmse, 2))

print("R² Score:", round(r2, 4))
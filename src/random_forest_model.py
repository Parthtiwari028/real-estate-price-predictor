import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load dataset
df = pd.read_csv("data/final_housing.csv")


# Features
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

# Target
y = df["price"]


# Categorical features
categorical_features = [
    "area_type",
    "location"
]

# Numerical features
numerical_features = [
    "total_sqft",
    "bath",
    "balcony",
    "bhk",
    "total_rooms"
]


# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            SimpleImputer(strategy="median"),
            numerical_features
        )
    ]
)


# Random Forest
rf_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    max_depth=25,
    min_samples_split=2
)


# Complete pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", rf_model)
    ]
)


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


# Train
print("\nTraining Random Forest...")
model.fit(X_train, y_train)


# Predict
y_pred = model.predict(X_test)


# Evaluation
mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)


print("\n--- RANDOM FOREST RESULTS ---")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R² Score:", round(r2, 4))


# Save model
joblib.dump(
    model,
    "models/real_estate_price_model.pkl"
)

print("\nModel saved successfully!")
print("Location: models/real_estate_price_model.pkl")
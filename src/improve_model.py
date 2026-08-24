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


# ==========================================
# 1. Load dataset
# ==========================================

df = pd.read_csv("data/final_housing.csv")

print("Original dataset:", df.shape)


# ==========================================
# 2. Remove extreme price-per-square-foot
# ==========================================

# Calculate price per square foot
df["price_per_sqft"] = (
    df["price"] * 100000 / df["total_sqft"]
)

# Calculate location-wise mean and standard deviation
location_stats = df.groupby("location")["price_per_sqft"].agg(
    ["mean", "std", "count"]
)


# ==========================================
# 3. Remove price outliers
# ==========================================

def remove_pps_outliers(data):

    excluded_indices = []

    for location, location_data in data.groupby("location"):

        mean = location_data.price_per_sqft.mean()
        std = location_data.price_per_sqft.std()

        # Keep values within 1 standard deviation
        filtered = location_data[
            (location_data.price_per_sqft >= mean - std) &
            (location_data.price_per_sqft <= mean + std)
        ]

        excluded_indices.extend(
            location_data.index.difference(filtered.index)
        )

    return data.drop(excluded_indices)


df = remove_pps_outliers(df)

print("After price-per-sqft outlier removal:", df.shape)


# ==========================================
# 4. Remove very large price outliers
# ==========================================

df = df[df["price"] < 1000]

print("After price filtering:", df.shape)


# ==========================================
# 5. Features
# ==========================================

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


# ==========================================
# 6. Preprocessing
# ==========================================

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


# ==========================================
# 7. Random Forest
# ==========================================

rf = RandomForestRegressor(
    n_estimators=250,
    max_depth=30,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)


model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", rf)
    ]
)


# ==========================================
# 8. Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


# ==========================================
# 9. Train
# ==========================================

print("\nTraining improved Random Forest...")

model.fit(X_train, y_train)


# ==========================================
# 10. Prediction
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 11. Evaluation
# ==========================================

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)


print("\n--- IMPROVED RANDOM FOREST ---")

print("MAE:", round(mae, 2))

print("RMSE:", round(rmse, 2))

print("R² Score:", round(r2, 4))


# ==========================================
# 12. Save model
# ==========================================

joblib.dump(
    model,
    "models/real_estate_price_model.pkl"
)

print("\nImproved model saved successfully!")
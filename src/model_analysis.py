import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==========================================
# Load dataset
# ==========================================

df = pd.read_csv("data/final_housing.csv")


# ==========================================
# Same outlier removal used during training
# ==========================================

df["price_per_sqft"] = (
    df["price"] * 100000 / df["total_sqft"]
)


def remove_pps_outliers(data):

    excluded_indices = []

    for location, location_data in data.groupby("location"):

        mean = location_data["price_per_sqft"].mean()
        std = location_data["price_per_sqft"].std()

        filtered = location_data[
            (location_data["price_per_sqft"] >= mean - std)
            &
            (location_data["price_per_sqft"] <= mean + std)
        ]

        excluded_indices.extend(
            location_data.index.difference(filtered.index)
        )

    return data.drop(excluded_indices)


df = remove_pps_outliers(df)

df = df[df["price"] < 1000]


# ==========================================
# Features
# ==========================================

features = [
    "area_type",
    "location",
    "total_sqft",
    "bath",
    "balcony",
    "bhk",
    "total_rooms"
]

X = df[features]
y = df["price"]


# ==========================================
# Train/Test split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================================
# Load trained model
# ==========================================

model = joblib.load(
    "models/real_estate_price_model.pkl"
)


# ==========================================
# Predictions
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# Model Metrics
# ==========================================

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)


print("\n================================")
print("       MODEL PERFORMANCE")
print("================================")

print("MAE :", round(mae, 2), "Lakhs")
print("RMSE:", round(rmse, 2), "Lakhs")
print("R²  :", round(r2, 4))


# ==========================================
# Actual vs Predicted
# ==========================================

plt.figure(figsize=(10, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.5
)

minimum = min(
    y_test.min(),
    y_pred.min()
)

maximum = max(
    y_test.max(),
    y_pred.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual Price (Lakhs)")
plt.ylabel("Predicted Price (Lakhs)")

plt.title(
    "Actual vs Predicted Property Prices"
)

plt.tight_layout()

plt.savefig(
    "actual_vs_predicted.png",
    dpi=300
)

plt.show()


print("\nGraph saved:")
print("actual_vs_predicted.png")


# ==========================================
# Feature Importance
# ==========================================

preprocessor = model.named_steps["preprocessor"]

rf = model.named_steps["model"]


# Get transformed feature names
feature_names = (
    preprocessor
    .get_feature_names_out()
)


importance = rf.feature_importances_


importance_df = pd.DataFrame(
    {
        "feature": feature_names,
        "importance": importance
    }
)


importance_df = importance_df.sort_values(
    "importance",
    ascending=False
)


print("\n================================")
print("       TOP FEATURES")
print("================================")

print(
    importance_df.head(15).to_string(index=False)
)


# ==========================================
# Feature Importance Graph
# ==========================================

top_features = importance_df.head(15)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["feature"][::-1],
    top_features["importance"][::-1]
)

plt.xlabel("Importance")

plt.ylabel("Feature")

plt.title(
    "Top 15 Features Influencing Property Price"
)

plt.tight_layout()

plt.savefig(
    "feature_importance.png",
    dpi=300
)

plt.show()


print("\nGraph saved:")
print("feature_importance.png")

print("\nModel analysis completed successfully!")

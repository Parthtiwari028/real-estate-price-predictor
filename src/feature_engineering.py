import pandas as pd

# Load cleaned dataset
df = pd.read_csv("data/cleaned_housing.csv")

print("Original shape:", df.shape)

# --------------------------------
# 1. Create Price Per Square Foot
# --------------------------------

df["price_per_sqft"] = (df["price"] * 100000) / df["total_sqft"]

# --------------------------------
# 2. Clean Location
# --------------------------------

df["location"] = df["location"].str.strip()

# Group locations with very few properties
location_counts = df["location"].value_counts()

rare_locations = location_counts[location_counts <= 10].index

df["location"] = df["location"].apply(
    lambda x: "Other" if x in rare_locations else x
)

# --------------------------------
# 3. Create Total Rooms
# --------------------------------

df["total_rooms"] = df["bhk"] + df["bath"]

# --------------------------------
# 4. Remove unrealistic values
# --------------------------------

# Minimum area per BHK
df = df[df["total_sqft"] / df["bhk"] >= 300]

# Remove extreme bathroom values
df = df[df["bath"] <= df["bhk"] + 2]

# --------------------------------
# 5. Remove unnecessary columns
# --------------------------------

df = df.drop("availability", axis=1)

# --------------------------------
# 6. Display information
# --------------------------------

print("\n--- FINAL COLUMNS ---")
print(df.columns.tolist())

print("\n--- FINAL SHAPE ---")
print(df.shape)

print("\n--- SAMPLE DATA ---")
print(df.head())

# --------------------------------
# 7. Save feature-engineered data
# --------------------------------

df.to_csv("data/final_housing.csv", index=False)

print("\nFeature engineering completed!")
print("Saved as: data/final_housing.csv")
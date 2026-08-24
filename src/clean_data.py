import pandas as pd

# Load dataset
df = pd.read_csv("data/Bengaluru_House_Data.csv")

print("Original shape:", df.shape)

# 1. Remove duplicate rows
df = df.drop_duplicates()

# 2. Drop society column
# It contains too many missing values.
df = df.drop("society", axis=1)

# 3. Remove rows with missing important values
df = df.dropna(subset=["location", "size", "bath"])

# 4. Fill missing balcony values with median
df["balcony"] = df["balcony"].fillna(df["balcony"].median())

# 5. Convert size into number of bedrooms
df["bhk"] = df["size"].str.extract("(\d+)").astype(float)

# 6. Convert total_sqft to numeric
def convert_sqft(x):
    try:
        if "-" in x:
            values = x.split("-")
            return (float(values[0]) + float(values[1])) / 2
        return float(x)
    except:
        return None

df["total_sqft"] = df["total_sqft"].apply(convert_sqft)

# 7. Remove rows where sqft conversion failed
df = df.dropna(subset=["total_sqft"])

# 8. Remove the original size column
df = df.drop("size", axis=1)

# Display result
print("Cleaned shape:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nFirst 5 rows:")
print(df.head())

# Save cleaned dataset
df.to_csv("data/cleaned_housing.csv", index=False)

print("\nCleaned dataset saved successfully!")
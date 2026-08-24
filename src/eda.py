import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("data/cleaned_housing.csv")

print("Dataset shape:", df.shape)

print("\n--- DATASET INFO ---")
print(df.info())

print("\n--- STATISTICS ---")
print(df.describe())

# -----------------------------
# 1. Price Distribution
# -----------------------------

plt.figure(figsize=(10, 6))

sns.histplot(df["price"], bins=50, kde=True)

plt.title("Distribution of Property Prices")
plt.xlabel("Price (Lakhs)")
plt.ylabel("Number of Properties")

plt.show()


# -----------------------------
# 2. Area vs Price
# -----------------------------

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="total_sqft",
    y="price"
)

plt.title("Property Area vs Price")
plt.xlabel("Total Area (sq.ft)")
plt.ylabel("Price (Lakhs)")

plt.show()


# -----------------------------
# 3. BHK vs Price
# -----------------------------

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df,
    x="bhk",
    y="price"
)

plt.title("BHK vs Property Price")
plt.xlabel("Number of Bedrooms")
plt.ylabel("Price (Lakhs)")

plt.show()


# -----------------------------
# 4. Average Price by Area Type
# -----------------------------

area_type_price = df.groupby("area_type")["price"].mean().sort_values()

print("\n--- AVERAGE PRICE BY AREA TYPE ---")
print(area_type_price)

plt.figure(figsize=(10, 6))

area_type_price.plot(kind="bar")

plt.title("Average Property Price by Area Type")
plt.xlabel("Area Type")
plt.ylabel("Average Price (Lakhs)")

plt.xticks(rotation=45)

plt.show()


# -----------------------------
# 5. Correlation
# -----------------------------

numeric_df = df.select_dtypes(include="number")

plt.figure(figsize=(8, 6))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.show()
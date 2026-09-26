import pandas as pd

# Load dataset
df = pd.read_csv("train.csv")

# 1. Inspect dataset
print("First 5 rows:")
print(df.head())

print("\nDataset information:")
df.info()

print("\nDataset shape:")
print(df.shape)

# 2. Check missing values
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# 3. Check duplicate records
print("\nDuplicate records:", df.duplicated().sum())

# 4. Remove duplicate records
df = df.drop_duplicates()

# 5. Handle missing Age values
df["Age"] = df["Age"].fillna(df["Age"].median())

# 6. Handle missing Embarked values
if "Embarked" in df.columns:
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# 7. Handle missing Cabin values
if "Cabin" in df.columns:
    df["Cabin"] = df["Cabin"].fillna("Unknown")

# 8. Check data types
print("\nData types:")
print(df.dtypes)

# 9. Check missing values after cleaning
print("\nMissing values after cleaning:")
print(df.isnull().sum())

# 10. Save cleaned dataset
df.to_csv("cleaned_train.csv", index=False)

print("\nCleaned dataset saved successfully!")
print("Final shape:", df.shape)
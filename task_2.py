import pandas as pd

# Load dataset
df = pd.read_csv("train.csv")

# 1. Basic information
print("Dataset Shape:")
print(df.shape)

print("\nDataset Information:")
df.info()

# 2. Descriptive statistics
print("\nDescriptive Statistics:")
print(df.describe())

# 3. Check unique values
print("\nUnique values:")
print(df.nunique())

# 4. Survival distribution
print("\nSurvival Count:")
print(df["Survived"].value_counts())

# 5. Survival percentage
print("\nSurvival Percentage:")
print(df["Survived"].value_counts(normalize=True) * 100)

# 6. Relationship between Gender and Survival
print("\nGender vs Survival:")
print(pd.crosstab(df["Sex"], df["Survived"]))

# 7. Relationship between Passenger Class and Survival
print("\nPassenger Class vs Survival:")
print(pd.crosstab(df["Pclass"], df["Survived"]))

# 8. Age statistics
print("\nAge Statistics:")
print(df["Age"].describe())

# 9. Detect outliers using IQR
Q1 = df["Age"].quantile(0.25)
Q3 = df["Age"].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

outliers = df[(df["Age"] < lower_limit) | (df["Age"] > upper_limit)]

print("\nAge Outliers:")
print(len(outliers))

# 10. Key findings
print("\nKey Findings:")

print("Total passengers:", len(df))

print("Survived:", df["Survived"].sum())

print("Did not survive:", (df["Survived"] == 0).sum())

print("Average age:", df["Age"].mean())

print("Average fare:", df["Fare"].mean())
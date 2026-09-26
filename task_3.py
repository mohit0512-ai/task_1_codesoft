import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("train.csv")

# 1. Survival Count Bar Chart
survival_count = df["Survived"].value_counts()

plt.figure(figsize=(6, 4))
survival_count.plot(kind="bar")
plt.title("Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")
plt.xticks(rotation=0)
plt.show()


# 2. Passenger Class Bar Chart
plt.figure(figsize=(6, 4))
df["Pclass"].value_counts().sort_index().plot(kind="bar")
plt.title("Passengers by Class")
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")
plt.xticks(rotation=0)
plt.show()


# 3. Age Distribution Histogram
plt.figure(figsize=(7, 4))
df["Age"].dropna().plot(kind="hist", bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")
plt.show()


# 4. Fare Distribution Histogram
plt.figure(figsize=(7, 4))
df["Fare"].plot(kind="hist", bins=20)
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Number of Passengers")
plt.show()


# 5. Gender vs Survival
gender_survival = pd.crosstab(df["Sex"], df["Survived"])

gender_survival.plot(kind="bar", figsize=(7, 4))
plt.title("Gender vs Survival")
plt.xlabel("Gender")
plt.ylabel("Number of Passengers")
plt.xticks(rotation=0)
plt.legend(["Did Not Survive", "Survived"])
plt.show()

print("Task 3 visualizations created successfully!")
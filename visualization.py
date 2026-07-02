import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("train.csv")

# Create output folder if it doesn't exist
import os
os.makedirs("output_images", exist_ok=True)

# 1. Survival Count
plt.figure(figsize=(5,4))
sns.countplot(x="Survived", data=df)
plt.title("Survival Count")
plt.savefig("output_images/survival_count.png")
plt.show()

# 2. Gender Count
plt.figure(figsize=(5,4))
sns.countplot(x="Sex", data=df)
plt.title("Gender Count")
plt.savefig("output_images/gender_count.png")
plt.show()

# 3. Age Distribution
plt.figure(figsize=(6,4))
plt.hist(df["Age"].dropna(), bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.savefig("output_images/age_distribution.png")
plt.show()

# 4. Fare Box Plot
plt.figure(figsize=(6,4))
sns.boxplot(y=df["Fare"])
plt.title("Fare Distribution")
plt.savefig("output_images/fare_boxplot.png")
plt.show()

# 5. Passenger Class Pie Chart
plt.figure(figsize=(6,6))
df["Pclass"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%"
)
plt.title("Passenger Class Distribution")
plt.ylabel("")
plt.savefig("output_images/passenger_class_pie.png")
plt.show()

# 6. Correlation Heatmap
plt.figure(figsize=(8,6))

numeric_df = df.select_dtypes(include=["number"])

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.savefig("output_images/correlation_heatmap.png")
plt.show()

print("All graphs generated successfully!")
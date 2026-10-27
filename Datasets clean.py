import pandas as pd
import numpy as np

# 1. Create a sample dataset
data = {
    "Student": ["Aarav", "Diya", "Kabir", "Meera", "Ishaan", "Diya"],
    "Department": ["AI&DS", "CSE", "AI&DS", "ECE", "CSE", "CSE"],
    "Age": [19, 20, 19, np.nan, 21, 20],
    "Python_Score": [85, 92, 78, 88, np.nan, 92],
    "Attendance": [90, 95, 82, 88, 76, 95]
}

df = pd.DataFrame(data)

# 2. Save and import CSV file
df.to_csv("students.csv", index=False)
df = pd.read_csv("students.csv")

print("Original Dataset:")
print(df)

# 3. Explore the dataset
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn information:")
df.info()

print("\nStatistical summary:")
print(df.describe())

# 4. Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# 5. Remove duplicate rows
df = df.drop_duplicates()

# 6. Handle missing numerical values
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Python_Score"] = df["Python_Score"].fillna(
    df["Python_Score"].median()
)

print("\nCleaned Dataset:")
print(df)

# 7. Select columns
print("\nStudent names and scores:")
print(df[["Student", "Python_Score"]])

# 8. Filter students scoring 85 or above
print("\nStudents scoring 85 or above:")
print(df[df["Python_Score"] >= 85])

# 9. Sort by Python score
print("\nSorted by Python score:")
print(df.sort_values("Python_Score", ascending=False))

# 10. Basic data analysis
print("\nAverage Python score:",
      round(df["Python_Score"].mean(), 2))

print("Highest score:", df["Python_Score"].max())
print("Lowest score:", df["Python_Score"].min())

print("Average attendance:",
      round(df["Attendance"].mean(), 2))

# 11. NumPy calculations
scores = df["Python_Score"].to_numpy()

print("\nNumPy mean:", np.mean(scores))
print("NumPy standard deviation:", np.std(scores))

# 12. Department-wise analysis
print("\nDepartment-wise average score:")
print(df.groupby("Department")["Python_Score"].mean())

# 13. Correlation
print("\nCorrelation matrix:")
print(df.select_dtypes(include=np.number).corr())

# 14. Export cleaned dataset
df.to_csv("cleaned_students.csv", index=False)

print("\nCleaned dataset saved successfully!")
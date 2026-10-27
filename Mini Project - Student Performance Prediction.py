# ============================================================
# MINI PROJECT: STUDENT PERFORMANCE PREDICTION
# Python + Pandas + NumPy + Matplotlib + Scikit-learn
# ============================================================

# STEP 1: Import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

print("Libraries imported successfully!")

# STEP 2: CREATE A SYNTHETIC DATASET

np.random.seed(42)

n = 500

study_hours = np.random.randint(1, 9, n)
attendance = np.random.randint(50, 101, n)
previous_scores = np.random.randint(35, 96, n)
sleep_hours = np.random.randint(4, 10, n)
tutoring_sessions = np.random.randint(0, 6, n)

parental_involvement = np.random.choice(
    ["Low", "Medium", "High"], n
)

internet_access = np.random.choice(
    ["Yes", "No"], n, p=[0.85, 0.15]
)

extracurricular = np.random.choice(
    ["Yes", "No"], n
)

# Generate scores using several factors plus random noise.
# This is illustrative synthetic data, not real student data.
noise = np.random.normal(0, 5, n)

exam_score = (
    15
    + 3.2 * study_hours
    + 0.20 * attendance
    + 0.25 * previous_scores
    + 1.2 * sleep_hours
    + 1.5 * tutoring_sessions
    + np.where(parental_involvement == "High", 4, 0)
    + np.where(parental_involvement == "Medium", 2, 0)
    + np.where(internet_access == "Yes", 2, 0)
    + np.where(extracurricular == "Yes", 1, 0)
    + noise
)

exam_score = np.clip(exam_score, 0, 100).round(1)

# Assemble the dataset
df = pd.DataFrame({
    "Hours_Studied": study_hours,
    "Attendance": attendance,
    "Previous_Scores": previous_scores,
    "Sleep_Hours": sleep_hours,
    "Tutoring_Sessions": tutoring_sessions,
    "Parental_Involvement": parental_involvement,
    "Internet_Access": internet_access,
    "Extracurricular_Activities": extracurricular,
    "Exam_Score": exam_score
})

# Add a few missing feature values for cleaning practice
for column in ["Sleep_Hours", "Previous_Scores"]:
    missing_indices = df.sample(
        frac=0.02, random_state=42
    ).index
    df.loc[missing_indices, column] = np.nan

# Save dataset
df.to_csv("student_performance.csv", index=False)

print("\nDataset created successfully!")
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

print("\nFirst 5 records:")
print(df.head())

print("\nDataset saved as student_performance.csv")

# STEP 3: EXPLORE THE DATASET

print("\nDataset information:")
df.info()

print("\nStatistical summary:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())

# STEP 4: CLEAN THE DATA

df = df.drop_duplicates().copy()

print("\nMissing values after cleaning will be handled")
print("inside the machine learning preprocessing pipeline.")

# STEP 5: VISUALIZE THE DATA

# Chart 1: Exam score distribution
plt.figure(figsize=(8, 5))
plt.hist(df["Exam_Score"], bins=20, edgecolor="black")
plt.title("Distribution of Student Exam Scores")
plt.xlabel("Exam Score")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.show()

# Chart 2: Study hours vs exam score
plt.figure(figsize=(8, 5))
plt.scatter(
    df["Hours_Studied"],
    df["Exam_Score"],
    alpha=0.6
)
plt.title("Study Hours vs Exam Score")
plt.xlabel("Hours Studied per Day")
plt.ylabel("Exam Score")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Chart 3: Attendance vs exam score
plt.figure(figsize=(8, 5))
plt.scatter(
    df["Attendance"],
    df["Exam_Score"],
    alpha=0.6
)
plt.title("Attendance vs Exam Score")
plt.xlabel("Attendance (%)")
plt.ylabel("Exam Score")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Chart 4: Average score by parental involvement
parent_scores = df.groupby(
    "Parental_Involvement"
)["Exam_Score"].mean()

plt.figure(figsize=(7, 5))
plt.bar(parent_scores.index, parent_scores.values)
plt.title("Average Score by Parental Involvement")
plt.xlabel("Parental Involvement")
plt.ylabel("Average Exam Score")
plt.tight_layout()
plt.show()

# STEP 6: PREPARE INPUTS AND TARGET

X = df.drop(columns=["Exam_Score"])
y = df["Exam_Score"]

numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

# Split before fitting any preprocessing steps
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# STEP 7: PREPROCESS MISSING VALUES AND CATEGORIES

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_transformer, numeric_features),
    ("categorical", categorical_transformer,
     categorical_features)
])

# STEP 8: BUILD TWO MACHINE LEARNING MODELS

models = {
    "Linear Regression": Pipeline([
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ]),

    "Random Forest": Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            min_samples_leaf=2
        ))
    ])
}

# STEP 9: TRAIN AND EVALUATE

results = {}
predictions = {}

for name, model in models.items():

    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    predictions[name] = pred

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    results[name] = {
        "MAE": mae,
        "RMSE": rmse,
        "R2_Score": r2
    }

results_df = pd.DataFrame(results).T

print("\nMODEL PERFORMANCE:")
print(results_df.round(3))

# STEP 10: COMPARE MODEL PERFORMANCE

results_df[["MAE", "RMSE"]].plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title("Model Error Comparison")
plt.ylabel("Error")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# STEP 11: ACTUAL VS PREDICTED SCORES

best_model_name = results_df["R2_Score"].idxmax()
best_model = models[best_model_name]
best_predictions = predictions[best_model_name]

plt.figure(figsize=(7, 6))
plt.scatter(y_test, best_predictions, alpha=0.7)

plt.plot(
    [0, 100],
    [0, 100],
    linestyle="--"
)

plt.title(f"Actual vs Predicted Scores: {best_model_name}")
plt.xlabel("Actual Exam Score")
plt.ylabel("Predicted Exam Score")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# STEP 12: PREDICT A NEW STUDENT'S SCORE

new_student = pd.DataFrame([{
    "Hours_Studied": 5,
    "Attendance": 85,
    "Previous_Scores": 75,
    "Sleep_Hours": 8,
    "Tutoring_Sessions": 2,
    "Parental_Involvement": "Medium",
    "Internet_Access": "Yes",
    "Extracurricular_Activities": "Yes"
}])

predicted_score = best_model.predict(new_student)[0]

print("\nNEW STUDENT PREDICTION")
print("----------------------")
print("Study hours:", new_student["Hours_Studied"].iloc[0])
print("Attendance:", new_student["Attendance"].iloc[0], "%")
print("Previous score:", new_student["Previous_Scores"].iloc[0])
print("Selected model:", best_model_name)
print("Predicted exam score:", round(predicted_score, 2))

# STEP 13: SAVE RESULTS

df.to_csv("student_performance.csv", index=False)

comparison_df = pd.DataFrame({
    "Actual_Score": y_test.to_numpy(),
    "Predicted_Score": np.round(best_predictions, 2)
})

comparison_df.to_csv(
    "student_score_predictions.csv",
    index=False
)

results_df.to_csv(
    "model_evaluation_results.csv"
)

print("\nPROJECT COMPLETED SUCCESSFULLY!")
print("Files created:")
print("1. student_performance.csv")
print("2. student_score_predictions.csv")
print("3. model_evaluation_results.csv")
# ============================================
# CARDIOVASCULAR DISEASE PREDICTION
# ============================================

# STEP 1: IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# ============================================
# STEP 2: LOAD DATASET
# ============================================

data = pd.read_csv("cardio_train (1).csv", sep=";")

print("\nDATASET LOADED SUCCESSFULLY!")

print("\nFIRST FIVE RECORDS")
print(data.head())

print("\nDATASET INFORMATION")
data.info()

print("\nDATASET SHAPE")
print(data.shape)

print("\nCOLUMN NAMES")
print(data.columns)

print("\nSTATISTICAL DESCRIPTION")
print(data.describe())


# ============================================
# STEP 3: DATA PREPROCESSING
# ============================================

print("\nMISSING VALUES")
print(data.isnull().sum())

print("\nDUPLICATE RECORDS")
print(data.duplicated().sum())

# Remove duplicate records
data = data.drop_duplicates()

# Remove rows containing missing values
data = data.dropna()

# Remove ID because it is only an identifier
data = data.drop("id", axis=1)

# Convert age from days into years
data["age"] = (data["age"] / 365.25).astype(int)

print("\nDATA CLEANING COMPLETED!")

print("\nCLEANED DATASET")
print(data.head())

print("\nNEW DATASET SHAPE")
print(data.shape)


# ============================================
# STEP 4: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================

# Heart Disease Distribution

print("\nHEART DISEASE DISTRIBUTION")
print(data["cardio"].value_counts())

plt.figure(figsize=(7, 5))

sns.countplot(data=data, x="cardio")

plt.title("Heart Disease Distribution")
plt.xlabel("Cardiovascular Disease (0 = No, 1 = Yes)")
plt.ylabel("Number of Patients")

plt.show()


# Age Distribution

plt.figure(figsize=(8, 5))

sns.histplot(data["age"], bins=30, kde=True)

plt.title("Age Distribution")
plt.xlabel("Age in Years")
plt.ylabel("Number of Patients")

plt.show()


# Gender Distribution

plt.figure(figsize=(8, 5))

sns.countplot(data=data, x="gender", hue="cardio")

plt.title("Heart Disease by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Patients")

plt.show()


# Cholesterol Distribution

plt.figure(figsize=(8, 5))

sns.countplot(data=data, x="cholesterol", hue="cardio")

plt.title("Heart Disease by Cholesterol")
plt.xlabel("Cholesterol Level")
plt.ylabel("Number of Patients")

plt.show()


# Glucose Distribution

plt.figure(figsize=(8, 5))

sns.countplot(data=data, x="gluc", hue="cardio")

plt.title("Heart Disease by Glucose")
plt.xlabel("Glucose Level")
plt.ylabel("Number of Patients")

plt.show()


# Blood Pressure Distribution

plt.figure(figsize=(8, 5))

sns.boxplot(data=data, x="cardio", y="ap_hi")

plt.title("Systolic Blood Pressure vs Heart Disease")

plt.show()


plt.figure(figsize=(8, 5))

sns.boxplot(data=data, x="cardio", y="ap_lo")

plt.title("Diastolic Blood Pressure vs Heart Disease")

plt.show()


# Weight Distribution

plt.figure(figsize=(8, 5))

sns.histplot(data["weight"], bins=30, kde=True)

plt.title("Weight Distribution")
plt.xlabel("Weight in KG")
plt.ylabel("Number of Patients")

plt.show()


# ============================================
# STEP 5: CORRELATION MATRIX
# ============================================

correlation = data.corr(numeric_only=True)

print("\nCORRELATION MATRIX")
print(correlation)

plt.figure(figsize=(12, 8))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Cardiovascular Disease Correlation Matrix")

plt.show()


# ============================================
# STEP 6: SPLIT FEATURES AND TARGET
# ============================================

X = data.drop("cardio", axis=1)

y = data["cardio"]

print("\nFEATURES")
print(X.columns)

print("\nTARGET")
print(y.name)


# ============================================
# STEP 7: TRAIN-TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTRAINING DATA SHAPE:", X_train.shape)

print("TESTING DATA SHAPE:", X_test.shape)


# ============================================
# STEP 8: FEATURE SCALING
# ============================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

print("\nFEATURE SCALING COMPLETED!")


# ============================================
# STEP 9: SVM MODEL
# ============================================

svm_model = SVC(kernel="linear")

svm_model.fit(X_train_scaled, y_train)

y_pred_svm = svm_model.predict(X_test_scaled)

svm_accuracy = accuracy_score(y_test, y_pred_svm)

print("\nSVM ACCURACY:", svm_accuracy * 100, "%")

print("\nSVM CLASSIFICATION REPORT")

print(classification_report(y_test, y_pred_svm))

cm_svm = confusion_matrix(y_test, y_pred_svm)

plt.figure(figsize=(6, 4))

sns.heatmap(cm_svm, annot=True, fmt="d", cmap="Blues")

plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()


# ============================================
# STEP 10: KNN MODEL
# ============================================

knn_model = KNeighborsClassifier(n_neighbors=5)

knn_model.fit(X_train_scaled, y_train)

y_pred_knn = knn_model.predict(X_test_scaled)

knn_accuracy = accuracy_score(y_test, y_pred_knn)

print("\nKNN ACCURACY:", knn_accuracy * 100, "%")

print("\nKNN CLASSIFICATION REPORT")

print(classification_report(y_test, y_pred_knn))

cm_knn = confusion_matrix(y_test, y_pred_knn)

plt.figure(figsize=(6, 4))

sns.heatmap(cm_knn, annot=True, fmt="d", cmap="Greens")

plt.title("KNN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()


# ============================================
# STEP 11: DECISION TREE MODEL
# ============================================

dt_model = DecisionTreeClassifier(
    criterion="gini",
    random_state=42
)

dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)

dt_accuracy = accuracy_score(y_test, y_pred_dt)

print("\nDECISION TREE ACCURACY:", dt_accuracy * 100, "%")

print("\nDECISION TREE CLASSIFICATION REPORT")

print(classification_report(y_test, y_pred_dt))

cm_dt = confusion_matrix(y_test, y_pred_dt)

plt.figure(figsize=(6, 4))

sns.heatmap(cm_dt, annot=True, fmt="d", cmap="Oranges")

plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()


# ============================================
# STEP 12: LOGISTIC REGRESSION
# ============================================

lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

lr_model.fit(X_train_scaled, y_train)

y_pred_lr = lr_model.predict(X_test_scaled)

lr_accuracy = accuracy_score(y_test, y_pred_lr)

print("\nLOGISTIC REGRESSION ACCURACY:", lr_accuracy * 100, "%")

print("\nLOGISTIC REGRESSION CLASSIFICATION REPORT")

print(classification_report(y_test, y_pred_lr))

cm_lr = confusion_matrix(y_test, y_pred_lr)

plt.figure(figsize=(6, 4))

sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Purples")

plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()


# ============================================
# STEP 13: RANDOM FOREST
# ============================================

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, y_pred_rf)

print("\nRANDOM FOREST ACCURACY:", rf_accuracy * 100, "%")

print("\nRANDOM FOREST CLASSIFICATION REPORT")

print(classification_report(y_test, y_pred_rf))

cm_rf = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(6, 4))

sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Reds")

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()


# ============================================
# STEP 14: ACCURACY COMPARISON
# ============================================

models = {
    "SVM": svm_accuracy,
    "KNN": knn_accuracy,
    "Decision Tree": dt_accuracy,
    "Logistic Regression": lr_accuracy,
    "Random Forest": rf_accuracy
}

comparison = pd.DataFrame(
    list(models.items()),
    columns=["Algorithm", "Accuracy"]
)

comparison["Accuracy"] = comparison["Accuracy"] * 100

comparison = comparison.sort_values(
    by="Accuracy",
    ascending=False
)

print("\nMODEL ACCURACY COMPARISON")

print(comparison.to_string(index=False))


# Accuracy Graph

plt.figure(figsize=(10, 6))

sns.barplot(
    data=comparison,
    x="Algorithm",
    y="Accuracy"
)

plt.title("Accuracy Comparison of ML Algorithms")

plt.xlabel("Algorithms")

plt.ylabel("Accuracy (%)")

plt.xticks(rotation=20)

plt.show()


# ============================================
# STEP 15: SELECT BEST MODEL
# ============================================

model_objects = {
    "SVM": svm_model,
    "KNN": knn_model,
    "Decision Tree": dt_model,
    "Logistic Regression": lr_model,
    "Random Forest": rf_model
}

best_algorithm = comparison.iloc[0]["Algorithm"]

best_model = model_objects[best_algorithm]

print("\nBEST PERFORMING MODEL:", best_algorithm)

print(
    "BEST MODEL ACCURACY:",
    comparison.iloc[0]["Accuracy"],
    "%"
)


# ============================================
# STEP 16: SAVE BEST MODEL
# ============================================

model_data = {
    "model": best_model,

    "scaler": scaler if best_algorithm in [
        "SVM",
        "KNN",
        "Logistic Regression"
    ] else None,

    "algorithm": best_algorithm,

    "features": list(X.columns)
}

joblib.dump(model_data, "cardio_best_model.pkl")

print("\n====================================")

print("BEST MODEL SAVED SUCCESSFULLY!")

print("MODEL NAME:", best_algorithm)

print("====================================")
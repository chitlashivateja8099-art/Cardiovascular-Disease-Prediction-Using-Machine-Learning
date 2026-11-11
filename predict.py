import pandas as pd
import joblib

# LOAD SAVED MODEL

model_data = joblib.load("cardio_best_model.pkl")

best_model = model_data["model"]

scaler = model_data["scaler"]

features = model_data["features"]

algorithm = model_data["algorithm"]


print("\n====================================")
print(" CARDIOVASCULAR DISEASE PREDICTION")
print("====================================")

print("\nSelected Algorithm:", algorithm)


# GET PATIENT DETAILS

age = float(input("\nEnter Age in Years: "))

gender = int(input("Enter Gender (1 or 2): "))

height = float(input("Enter Height in cm: "))

weight = float(input("Enter Weight in kg: "))

ap_hi = float(input("Enter Systolic Blood Pressure: "))

ap_lo = float(input("Enter Diastolic Blood Pressure: "))

cholesterol = int(input("Enter Cholesterol (1, 2 or 3): "))

gluc = int(input("Enter Glucose (1, 2 or 3): "))

smoke = int(input("Do you smoke? (0 = No, 1 = Yes): "))

alco = int(input("Consume alcohol? (0 = No, 1 = Yes): "))

active = int(input("Physically active? (0 = No, 1 = Yes): "))


# CREATE PATIENT DATAFRAME

patient = {
    "age": age,
    "gender": gender,
    "height": height,
    "weight": weight,
    "ap_hi": ap_hi,
    "ap_lo": ap_lo,
    "cholesterol": cholesterol,
    "gluc": gluc,
    "smoke": smoke,
    "alco": alco,
    "active": active
}

patient_data = pd.DataFrame(
    [[patient[column] for column in features]],
    columns=features
)


# FEATURE SCALING

if scaler is not None:

    patient_data = scaler.transform(patient_data)


# PREDICTION

prediction = best_model.predict(patient_data)


# DISPLAY RESULT

print("\n====================================")
print("          PREDICTION RESULT")
print("====================================")

if prediction[0] == 1:

    print("Predicted Class: 1")

    print("Cardiovascular Disease Class")

else:

    print("Predicted Class: 0")

    print("No Cardiovascular Disease Class")

print("\nPREDICTION COMPLETED!")
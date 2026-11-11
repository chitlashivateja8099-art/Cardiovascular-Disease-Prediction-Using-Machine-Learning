import tkinter as tk
from tkinter import ttk, messagebox

import pandas as pd
import joblib


# =================================
# LOAD SAVED MODEL
# =================================

model_data = joblib.load("cardio_best_model.pkl")

best_model = model_data["model"]

scaler = model_data["scaler"]

algorithm = model_data["algorithm"]

features = model_data["features"]


# =================================
# CREATE MAIN WINDOW
# =================================

root = tk.Tk()

root.title("Cardiovascular Disease Prediction")

root.geometry("600x750")

root.configure(bg="#EAF2F8")


# =================================
# PROJECT TITLE
# =================================

title = tk.Label(
    root,
    text="CARDIOVASCULAR DISEASE PREDICTION",
    font=("Arial", 17, "bold"),
    bg="#EAF2F8",
    fg="#154360"
)

title.pack(pady=20)


subtitle = tk.Label(
    root,
    text="Machine Learning Based Prediction System",
    font=("Arial", 11),
    bg="#EAF2F8"
)

subtitle.pack()


# =================================
# INPUT FRAME
# =================================

frame = tk.Frame(
    root,
    bg="white",
    padx=25,
    pady=20
)

frame.pack(pady=20)


inputs = {}


# =================================
# ENTRY FUNCTION
# =================================

def create_entry(label_text, key, row):

    label = tk.Label(
        frame,
        text=label_text,
        font=("Arial", 11),
        bg="white"
    )

    label.grid(
        row=row,
        column=0,
        padx=10,
        pady=8,
        sticky="w"
    )

    entry = tk.Entry(
        frame,
        font=("Arial", 11),
        width=20
    )

    entry.grid(
        row=row,
        column=1,
        padx=10,
        pady=8
    )

    inputs[key] = entry


# =================================
# DROPDOWN FUNCTION
# =================================

def create_dropdown(label_text, key, values, row):

    label = tk.Label(
        frame,
        text=label_text,
        font=("Arial", 11),
        bg="white"
    )

    label.grid(
        row=row,
        column=0,
        padx=10,
        pady=8,
        sticky="w"
    )

    dropdown = ttk.Combobox(
        frame,
        values=values,
        state="readonly",
        width=18,
        font=("Arial", 11)
    )

    dropdown.grid(
        row=row,
        column=1,
        padx=10,
        pady=8
    )

    inputs[key] = dropdown


# =================================
# PATIENT DETAILS
# =================================

create_entry("Age (Years)", "age", 0)

create_dropdown("Gender", "gender", [1, 2], 1)

create_entry("Height (cm)", "height", 2)

create_entry("Weight (kg)", "weight", 3)

create_entry("Systolic BP", "ap_hi", 4)

create_entry("Diastolic BP", "ap_lo", 5)

create_dropdown("Cholesterol", "cholesterol", [1, 2, 3], 6)

create_dropdown("Glucose", "gluc", [1, 2, 3], 7)

create_dropdown("Smoking", "smoke", [0, 1], 8)

create_dropdown("Alcohol", "alco", [0, 1], 9)

create_dropdown("Physical Activity", "active", [0, 1], 10)


# =================================
# PREDICTION FUNCTION
# =================================

def predict_disease():

    try:

        patient = {}

        # Collect input values

        for key, widget in inputs.items():

            value = widget.get()

            if value == "":

                messagebox.showerror(
                    "Input Error",
                    "Please fill in all details."
                )

                return

            patient[key] = float(value)


        # Basic validation

        if not 0 < patient["age"] <= 120:

            messagebox.showerror(
                "Input Error",
                "Enter a valid age."
            )

            return

        if not 50 <= patient["height"] <= 250:

            messagebox.showerror(
                "Input Error",
                "Enter a valid height in cm."
            )

            return

        if not 1 <= patient["weight"] <= 400:

            messagebox.showerror(
                "Input Error",
                "Enter a valid weight in kg."
            )

            return

        if patient["gender"] not in [1, 2]:

            raise ValueError("Invalid gender value.")

        if patient["cholesterol"] not in [1, 2, 3]:

            raise ValueError("Invalid cholesterol value.")

        if patient["gluc"] not in [1, 2, 3]:

            raise ValueError("Invalid glucose value.")


        # Create DataFrame

        patient_data = pd.DataFrame(
            [[patient[column] for column in features]],
            columns=features
        )


        # Apply scaling

        if scaler is not None:

            patient_data = scaler.transform(patient_data)


        # Make prediction

        prediction = best_model.predict(patient_data)


        # Display result

        if prediction[0] == 1:

            result_label.config(
                text="Predicted Class: 1\nCardiovascular Disease",
                fg="red"
            )

        else:

            result_label.config(
                text="Predicted Class: 0\nNo Cardiovascular Disease",
                fg="green"
            )


    except ValueError as error:

        messagebox.showerror(
            "Input Error",
            str(error)
        )

    except Exception as error:

        messagebox.showerror(
            "Error",
            str(error)
        )


# =================================
# PREDICT BUTTON
# =================================

predict_button = tk.Button(
    root,
    text="PREDICT",
    font=("Arial", 13, "bold"),
    bg="#2874A6",
    fg="white",
    width=20,
    command=predict_disease
)

predict_button.pack(pady=10)


# =================================
# RESULT DISPLAY
# =================================

result_label = tk.Label(
    root,
    text="Enter patient details and click Predict",
    font=("Arial", 13, "bold"),
    bg="#EAF2F8",
    fg="#154360"
)

result_label.pack(pady=15)


# =================================
# MODEL INFORMATION
# =================================

algorithm_label = tk.Label(
    root,
    text="Selected Algorithm: " + algorithm,
    font=("Arial", 10),
    bg="#EAF2F8"
)

algorithm_label.pack(pady=5)


# =================================
# RUN APPLICATION
# =================================

root.mainloop()
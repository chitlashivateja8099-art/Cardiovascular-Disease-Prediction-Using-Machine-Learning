# Cardiovascular Disease Prediction Using Machine Learning

## Project Overview

This project uses Machine Learning algorithms to predict the likelihood of cardiovascular disease based on patient health parameters.

It analyzes medical attributes such as age, gender, height, weight, blood pressure, cholesterol, glucose, smoking, alcohol consumption, and physical activity.

Five machine learning algorithms are implemented and compared to identify the best-performing model.

## Objectives

- Predict cardiovascular disease using machine learning.
- Perform data preprocessing and exploratory data analysis (EDA).
- Train and evaluate multiple classification algorithms.
- Compare model performance using accuracy and classification reports.
- Develop a user-friendly GUI for predictions.

##Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Tkinter
- Joblib

## Machine Learning Algorithms

The following five algorithms are implemented:

1. Support Vector Machine (SVM)
2. Logistic Regression
3. Random Forest
4. K-Nearest Neighbors (KNN)
5. Decision Tree

## Model Accuracy Comparison

| Algorithm | Accuracy |
|---|---:|
| SVM | 71.99% |
| Logistic Regression | 71.39% |
| Random Forest | 70.56% |
| KNN | 64.61% |
| Decision Tree | 63.38% |

**Best Performing Model: SVM**

**Accuracy: 71.99%**

## Project Structure

```text
Cardiovascular-Disease-Prediction/
│
├── cardio_train (1).csv
├── main.py
├── predict.py
├── gui.py
├── cardio_best_model.pkl
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the Project Folder

```bash
cd Cardiovascular-Disease-Prediction
```

### 3. Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

## How to Run the Project

### Step 1: Train the Models

```bash
python main.py
```

This performs data preprocessing, visualization, model training, and accuracy comparison. The best-performing model is saved as `cardio_best_model.pkl`.

### Step 2: Predict Cardiovascular Disease

```bash
python predict.py
```

Enter the requested patient health details to receive a prediction.

### Step 3: Launch the GUI

```bash
python gui.py
```

The graphical interface allows users to enter patient details and obtain predictions.

## Project Workflow

1. Dataset Collection
2. Data Loading
3. Data Preprocessing
4. Exploratory Data Analysis
5. Feature Selection
6. Train-Test Splitting
7. Feature Scaling
8. Model Training
9. Model Evaluation
10. Best Model Selection
11. Prediction
12. GUI Development

## Future Improvements

- Improve model performance through hyperparameter tuning.
- Explore additional machine learning algorithms.
- Improve the graphical user interface.
- Add more evaluation metrics and visualizations.
- Deploy the application as a web-based prediction system.

## ⚠️ Disclaimer

This project is developed for educational and learning purposes only. Its predictions are not medical diagnoses and should not be used for clinical decisions. Always consult qualified healthcare professionals for medical advice.

## 👨‍💻 Author

**CHITLA SHIVA TEJA REDDY**

B.Tech – Artificial Intelligence and Machine Learning

REVA University

## ⭐ Acknowledgement

This project helped me understand data preprocessing, exploratory data analysis, machine learning classification algorithms, model evaluation, and GUI development using Python.

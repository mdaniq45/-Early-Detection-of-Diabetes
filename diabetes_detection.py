
import pandas as pd

# Load the dataset
df = pd.read_csv(r"diabetes.csv")  # Replace with your actual file path

# Display basic information
print(df.info())
print(df.head())  # Show the first 5 rows
print(df.isnull().sum())  # Check missing values
import numpy as np

# Replace zero values with the mean of each column
cols_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

for col in cols_to_replace:
    df[col] = df[col].replace(0, np.mean(df[col]))

print(df.describe())  # Check data after replacement
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = df.drop(columns=['Outcome'])  # Independent variables
y = df['Outcome']  # Target variable

X_scaled = scaler.fit_transform(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Train the model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

# Predict
y_pred = lr_model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Logistic Regression Accuracy: {accuracy:.2f}")
from sklearn.svm import SVC

svm_model = SVC()
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

print(f"SVM Accuracy: {accuracy_score(y_test, y_pred_svm):.2f}")
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf):.2f}")
from sklearn.metrics import classification_report, confusion_matrix

print("Logistic Regression Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
import joblib

joblib.dump(rf_model, "diabetes_model.pkl")  # Save the trained model
print("Model saved as 'diabetes_model.pkl'")
# Load the model
loaded_model = joblib.load("diabetes_model.pkl")

# Example input for prediction
new_data = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])  # Replace with user input
new_data_scaled = scaler.transform(new_data)  # Scale input

# Make prediction
prediction = loaded_model.predict(new_data_scaled)
print("Prediction:", "Diabetic" if prediction[0] == 1 else "Not Diabetic")
import streamlit as st

# Load model
model = joblib.load("diabetes_model.pkl")

st.title("🦠 Early Detection of Diabetes")

# User inputs
pregnancies = st.number_input("Pregnancies", 0, 20)
glucose = st.number_input("Glucose Level", 0, 200)
bp = st.number_input("Blood Pressure", 0, 150)
skin_thickness = st.number_input("Skin Thickness", 0, 100)
insulin = st.number_input("Insulin", 0, 900)
bmi = st.number_input("BMI", 0.0, 50.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5)
age = st.number_input("Age", 1, 100)

if st.button("Predict"):
    # Prepare input
    user_input = np.array([[pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]])
    user_input_scaled = scaler.transform(user_input)
    
    # Prediction
    prediction = model.predict(user_input_scaled)
    
    st.success("🟢 Not Diabetic" if prediction[0] == 0 else "🔴 Diabetic")

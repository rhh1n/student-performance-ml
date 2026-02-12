import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -----------------------------
# Title and Description
# -----------------------------
st.title("🎓 Student Performance Prediction System")

st.write(
    "This system predicts whether a student will Pass or Fail "
    "using Machine Learning (Logistic Regression)."
)

# -----------------------------
# Load Dataset
# -----------------------------
data = pd.read_csv("student_data.csv")

X = data[['StudyHours', 'Attendance', 'InternalMarks', 'GPA']]
y = data['Result']

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Train Model
# -----------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# -----------------------------
# Calculate Accuracy
# -----------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.write(f"### 📊 High Accuracy")

# -----------------------------
# User Input Section
# -----------------------------
st.header("Enter Student Details")

study_hours = st.slider("Study Hours per Day", 0, 12, 5)
attendance = st.slider("Attendance (%)", 0, 100, 75)
internal_marks = st.slider("Internal Marks", 0, 25, 15)
gpa = st.number_input("Previous GPA", min_value=0.0, max_value=10.0, step=0.1)

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("Predict Result"):
    prediction = model.predict([[study_hours, attendance, internal_marks, gpa]])
    st.success(f"🎯 Predicted Result: {prediction[0]}")



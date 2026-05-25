import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load dataset
data = pd.read_csv("student_data.csv")

# Encode text columns
le = LabelEncoder()

data['Gender'] = le.fit_transform(data['Gender'])
data['Degree'] = le.fit_transform(data['Degree'])
data['Branch'] = le.fit_transform(data['Branch'])
data['Placement_Status'] = le.fit_transform(data['Placement_Status'])

# Features
X = data[['CGPA',
          'Internships',
          'Projects',
          'Coding_Skills',
          'Communication_Skills',
          'Aptitude_Test_Score',
          'Soft_Skills_Rating']]

# Target
y = data['Placement_Status']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Streamlit UI
st.title("Student Placement Prediction System")

cgpa = st.number_input("Enter CGPA", min_value=0.0, max_value=10.0)

internships = st.number_input("Enter number of internships", min_value=0)

projects = st.number_input("Enter number of projects", min_value=0)

coding = st.slider("Coding Skills", 1, 10)

communication = st.slider("Communication Skills", 1, 10)

aptitude = st.number_input("Aptitude Test Score", min_value=0)

softskills = st.slider("Soft Skills Rating", 1, 10)

# Prediction button
if st.button("Predict Placement"):

    prediction = model.predict([[cgpa,
                                 internships,
                                 projects,
                                 coding,
                                 communication,
                                 aptitude,
                                 softskills]])

    if prediction[0] == 1:
        st.success("Student is likely to get placement")
    else:
        st.error("Student may not get placement")
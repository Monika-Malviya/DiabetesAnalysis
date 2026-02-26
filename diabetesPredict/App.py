import numpy as np
import pandas as pd
import joblib 
import streamlit as st



st.title("DiabetesPrediction Application")
st.header("Machine Learning")

model = joblib.load("DiabetesPrediction.pkl")
# model = pickle.load("DiabetesPrediction.pkl")
# with open("DiabetesPrediction.pkl", "rb") as file:
#     model = pickle.load(file)

# Pregnancies	
# Glucose	BloodPressure	SkinThickness	Insulin	BMI	DiabetesPedigreeFunction	Age

Pregnancies = st.number_input("Pregnancies ", min_value=0 , max_value= 17)
Glucose = st.number_input("Glucose ", min_value=0 , max_value= 200)
BloodPressure = st.number_input("BloodPressure ", min_value=0 , max_value= 125)
SkinThickness = st.number_input("SkinThickness ", min_value=0 , max_value= 100)
Insulin = st.number_input("Insulin ", min_value=0 , max_value= 850)
BMI = st.number_input("BMI ", min_value=0.0 , max_value= 70.0)
DiabetesPedigreeFunction = st.number_input("DiabetesPedigreeFunction ", min_value=0.0 , max_value= 3.0)
Age = st.number_input("Age ", min_value=18 , max_value= 85)



if st.button("predict"):
    new_array = np.array([[Pregnancies,Glucose	,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
    prediction = model.predict(new_array)

    if prediction == 0:
        st.write("Negative")
    else:
        st.write("Positive")    
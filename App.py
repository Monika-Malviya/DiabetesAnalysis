import numpy as np
import pandas as pd
import pickle
import streamlit as st

st.title("Diabetes Prediction Application")
st.header("Machine learning Project")

with open("DiabetesPrediction.pkl",'rb') as file:
    model = pickle.load(file)

# Pregnancies	
# Glucose	BloodPressure	SkinThickness	Insulin	BMI	DiabetesPedigreeFunction	Age

Pregnancy = st.number_input("Pregnancies",min_value=0,max_value=17)
Glucose = st.number_input("Glucose",min_value=0,max_value=200)
BloodPressure = st.number_input("Blood Pressure",min_value=0,max_value=125)
SkinThickness = st.number_input("SkinThickness",min_value=0,max_value=100)
Insulin = st.number_input("Insulin",min_value=0,max_value=850)
BMI = st.number_input("BMI",min_value=0.0,max_value=70.0)
DiabetesPedigreeFunction = st.number_input("DiabetesPedigreeFunction",min_value=0.0,max_value=3.0)
Age = st.number_input("Age",min_value=18, max_value=85)

if st.button("Predict"):
    new_array = np.array([[Pregnancy, Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
    prediction = model.predict(new_array)
    
    if prediction == 0:
        st.success("Negative")  
    else:
        st.success("Positive")
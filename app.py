# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 11:13:53 2021

@author: Pertz
"""
import streamlit as st
import pandas as pd
import pickle
from PIL import Image
model = pickle.load(open('RandomForest.pkl', 'rb'))

st.header("Heart Disease Classification:")
image = Image.open('pertz.jpg')
st.image(image, use_column_width=True)
st.write("Please insert values, to get prediction")

Age = st.slider('Age:', 10.0, 120.0)
Sex = st.slider('Sex:', 0.0, 1.0)
ChsetPain = st.slider('Chest Pain Type:', 1.0, 4.0)
RestingBP = st.slider('Resting BP:', 90.0, 200.0)
Cholesterol = st.slider('Cholesterol:', 100.0, 300.0)
FastingBS = st.slider('Fasting Blood Sugr:', 0.0, 1.0)
RestingECG = st.slider('Resting ECG:', 0.0, 2.0)
MaxHR = st.slider('Maximum Heart Rate:', 85.0, 200.0)
ExAngina = st.slider('Exercise Angina:', 0.0, 1.0)
OldPeak = st.slider('Old Peak:', 0.0, 5.0)
STslope = st.slider('ST Slope:', 0.0, 1.0)
data = {'age': Age,
        'sex': Sex,
        'chest pain type': ChsetPain,
        'resting bp s': RestingBP,
        'cholesterol': Cholesterol,
        'fasting blood sugar':FastingBS,
        'resting ecg':RestingECG,
        'max heart rate':MaxHR,
        'exercise angina': ExAngina,
        'oldpeak': OldPeak,
        'ST slope': STslope       
        }

features = pd.DataFrame(data, index=[0])

pred_proba = model.predict_proba(features)
#or
prediction = model.predict(features)

st.subheader('Prediction Percentages:') 
st.write('**Probablity of not havig a cardiovascular disease is ( in % )**:',pred_proba[0][0]*100)
st.write('**Probablity of havig a cardiovascular disease is ( in % )**:',pred_proba[0][1]*100)

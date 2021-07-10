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

st.header("Heart Disease Prediction:")
image = Image.open('pertz.jpg')
st.image(image, use_column_width=True)
st.write("Please insert values, to get prediction")

Age = st.slider('Age:', 10, 120)
Sex = st.radio("Select Gender:", (1,0))
if (Sex == 1):
    st.success("Male")
else:
    st.success("Female")
ChsetPain = st.slider('Chest Pain Type:', 1, 4)
if (ChsetPain == 1):
    st.success("Typical Angina")
elif (ChsetPain == 2):
    st.success("Atypical Angina")
elif (ChsetPain == 3):
    st.success("Non-anginal Pain")
else:
    st.success("Asymptomatic")
RestingBP = st.slider('Resting BP:', 90, 200)
Cholesterol = st.slider('Cholesterol (serum cholestoral in mg/dl):', 100, 300)
FastingBS = st.slider('Fasting Blood Sugar > 120 mg/dl) (1 = true; 0 = false):', 0, 1)
RestingECG = st.slider('Resting Electrocardiographic Results:', 0, 2)
if (RestingECG  == 0):
    st.success(" Normal")
elif (RestingECG == 1):
    st.success("Have ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)")
else:
    st.success(" showing probable or definite left ventricular hypertrophy by Estes' criteria")
MaxHR = st.slider('Maximum Heart Rate:', 85, 200)
ExAngina = st.slider('Exercise Induced Angina(1 = yes; 0 = no):', 0, 1)
OldPeak = st.slider('Old Peak:', 0, 5)
STslope = st.slider('ST Slope:', 0, 1)
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

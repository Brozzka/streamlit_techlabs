import streamlit as st
from xgboost import XGBClassifier
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

root_dir = os.path.dirname(__file__)
#features_dir = os.path.join(root_dir, 'features')
csv_file = os.path.join(root_dir, 'alzheimer.csv')
data=pd.read_csv(csv_file)
data["SES"].fillna(data["SES"].mean(), inplace=True)
data["MMSE"].fillna(data["MMSE"].mean(), inplace=True)
data = data[data["Group"]!= "Converted"]
data['Group'].replace(['Nondemented', 'Demented'],[0, 1], inplace=True)
data['M/F'].replace(['M', 'F'],[0, 1], inplace=True)



st.header("Classification of Demented/NonDemented using Alzheimer feature dataset")
st.subheader("Predicts the diagnosis of Alzheimer's disease based on the subjects personal data")
st.write("This application uses XGBoost")

#features = st.beta_container()
predictor = st.container()    

with predictor:
    sel = st.selectbox('Pick the gender of the subject', ['Male','Female'])
    M_F = 0
    if sel == 'Female':
        M_F = 1
    Age = st.slider('Age of the subject ', 60, 98,step = 1)
    Age = (Age - data['Age'].min())/(data['Age'].max() - data['Age'].min()) # normalizing

    EDUC = st.slider('Years of Education of the subject ', 6, 23,step = 1)
    EDUC = (EDUC - data['EDUC'].min())/(data['EDUC'].max() - data['EDUC'].min()) # normalizing

    SES = st.slider('Socioeconomic Status of the subject ', 1, 5,step = 1)
    SES = (SES - data['SES'].min())/(data['SES'].max() - data['SES'].min()) # normalizing

    MMSE = st.slider('Mini Mental State Examination of the subject ', 4, 30,step = 1)
    MMSE = (MMSE - data['MMSE'].min())/(data['MMSE'].max() - data['MMSE'].min()) # normalizing
   
    CDR = st.slider('Clinical Dementia Rating of the subject ', 0, 3,step = 1)
    CDR = (CDR - data['CDR'].min())/(data['CDR'].max() - data['CDR'].min()) # normalizing

    eTIV = st.slider('Estimated total intracranial volume of the subject ', 1106, 2004)
    eTIV = (eTIV - data['eTIV'].min())/(data['eTIV'].max() - data['eTIV'].min()) # normalizing   

    nWBV = st.slider('Normalized Whole Brain Volume of the subject ', 0.644, 0.837)
    nWBV = (nWBV - data['nWBV'].min())/(data['nWBV'].max() - data['nWBV'].min()) # normalizing

    ASF = st.slider('Atlas Scaling Factor of the subject ', 0.876, 1.587)
    ASF = (ASF - data['ASF'].min())/(data['ASF'].max() - data['ASF'].min()) # normalizing

    xTest = np.array([M_F , Age , EDUC, SES, MMSE, CDR, eTIV, nWBV ,ASF])
    #st.text(xTest)
    xTest =np.expand_dims(xTest, axis=0)
    xgb_model = XGBClassifier()
    xgb_model.load_model(os.path.join(root_dir,"model.json"))
    
    y_pred=xgb_model.predict(xTest)[0]
    #st.text((y_pred))
    class_names = ['Nondemented', 'Demented']
    a = class_names[y_pred]
    string = "The subject is predicted to be: " + a
    st.success(string)
#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import streamlit as st
from xgboost import XGBRegressor


# In[3]:


st.title('Energy Production')
st.sidebar.header('Enter the values')


# In[15]:


def user_defined_values():
    temperature= st.sidebar.number_input('Temperature')
    exhaust_vacuum= st.sidebar.number_input('Exhaust Vaccum')
    amb_pressure= st.sidebar.number_input('Ambient Pressure')
    r_humidity= st.sidebar.number_input('Relative Humidity')
    data={'temperature':temperature,'exhaust_vacuum':exhaust_vacuum,'amb_pressure':amb_pressure,'r_humidity':r_humidity}
    features= pd.DataFrame(data,index=[0])
    return features
df= user_defined_values()
st.subheader('Let us predict the energy production')
st.write()
energy=pd.read_csv(r'C:\Users\my computer\Documents\Data Science\Projects\P324\Regrerssion_energy_production_data.csv',delimiter=';')
erergy=energy.drop_duplicates(inplace=True)
X=energy.drop('energy_production',axis=1)
Y=energy[['energy_production']]
model=XGBRegressor(n_estimators=250,learning_rate=0.2, max_depth=5)
model.fit(X,Y)
if st.button('Predict'):
    predicted_value = model.predict(df)
    st.subheader('Predicted Value:')
    st.write(predicted_value[0])


# In[ ]:





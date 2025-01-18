import streamlit as st
import numpy as np
import pickle
import joblib
from  tensorflow.keras.models import load_model

import   streamlit  as st; from PIL import Image; import numpy  as np
import pandas  as pd; import pickle

import os

filename1 = 'https://raw.githubusercontent.com/imsb1371/Geopoly/refs/heads/main/Capture1.PNG'
filename2 = 'https://raw.githubusercontent.com/imsb1371/Geopoly/refs/heads/main/Capture2.PNG'

st.title('Heavy metal immobilization in the solidification/stabilization of municipal solid waste incineration fly ash using geopolymers')
with st.container():
    st.image(filename1)
    st.image(filename2)


# Arrange input boxes for new parameters
col1, col2, col3 = st.columns(3)
with col1:
    Si_Al = st.number_input('Si/Al Ratio', 0.0)
with col2:
    Si_Ca = st.number_input('Si/Ca Ratio', 0.0)
with col3:
    Fe = st.number_input('Fe', 0.0)

col4, col5, col6 = st.columns(3)
with col4:
    T = st.number_input('Temperature', 0.0)
with col5:
    CT = st.number_input('CT', 0.0)
with col6:
    L_S = st.number_input('L/S Ratio', 0.0)

col7, col8, col9 = st.columns(3)
with col7:
    AE = st.number_input('AE', 0.0)
with col8:
    AM = st.number_input('AM', 0.0)
with col9:
    HMV = st.number_input('HMV', 0.0)

col10, col11, col12 = st.columns(3)
with col10:
    IC = st.number_input('IC', 0.0)
with col11:
    HME = st.number_input('HME', 0.0)
with col12:
    RHM = st.number_input('RHM', 0.0)

# Gather all inputs into a list for normalization
input_values = [Si_Al, Si_Ca, Fe, T, CT, L_S, AE, AM, HMV, IC, HME, RHM]

# Define min and max values for normalization
min_values = [3.22, 0.06, 0.65, 20, 3, 0.42, 7, 1.55, 2, 0.115, 0.89, 4.01]
max_values = [12.51, 21.71, 3.535, 80, 28, 0.67, 11, 3.16, 3, 280.4375, 2.33, 4.61]

# Normalize the input values based on min and max values
normalized_inputs = [
    (2 * (val - min_val) / (max_val - min_val) - 1)
    for val, min_val, max_val in zip(input_values, min_values, max_values)
]

# Convert normalized inputs to a numpy array
inputvec = np.array(normalized_inputs)

# Check for zeros in the numeric features
zero_count = sum(1 for value in input_values if value == 0)

if st.button('Run'):
    if zero_count > 1:
        st.error("Error: More than one input value is zero. Please provide valid inputs for features.")
    else:
        try:
            # Load the model
            model2 = joblib.load('Model.pkl')

            # Predict using the model
            inputvec = inputvec.reshape(1, -1)  # Ensure correct shape
            YY = model2.predict(inputvec)

            # Calculate removal efficiency
            RE = (YY + 1) * (100.0 - 0.0) * 0.5 + 0.0
            RE = min(RE, 99)  # Limit RE to 99%

            # Display predictions
            st.write("Removal efficiency (%): ", np.round(abs(RE), 2))

        except Exception as e:
            st.error(f"Model prediction failed: {e}")


filename7 = 'https://raw.githubusercontent.com/imsb1371/Geopoly/refs/heads/main/Capture3.PNG'
filename8 = 'https://raw.githubusercontent.com/imsb1371/Geopoly/refs/heads/main/Capture4.PNG'

col22, col23 = st.columns(2)
with col22:
    with st.container():
        st.markdown("<h5>Developer:</h5>", unsafe_allow_html=True)
        st.image(filename8)

with col23:
    with st.container():
        st.markdown("<h5>Supervisor:</h5>", unsafe_allow_html=True)
        st.image(filename7) 


footer = """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f1f1f1;
        color: black;
        text-align: center;
        padding: 10px;
        font-size: 12px;
    }
    </style>
    <div class="footer">
    This web app was developed in School of Resources and Safety Engineering, Central South University, Changsha 410083, China
    </div>
"""
st.markdown(footer, unsafe_allow_html=True)

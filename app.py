import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json

pickle_in = open('bangalore_house_price_model.pickle','rb')
regressor = pickle.load(pickle_in)

with open('columns.json','r') as f:
    data = json.load(f)

locations = [locs for locs in data['data_columns'][3:]]

columns = [cols for cols in data['data_columns']]
X = pd.DataFrame(columns = columns)
def predict_price(location,sqft,bath,bhk):
    loc_index = np.where(X.columns==location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return regressor.predict([x])[0]

st.title('House Price Prediction Model')
location = st.selectbox('Select a location',locations)
sq_ft = st.number_input('Enter the square feet')
bedrooms = st.number_input('Enter number of bed rooms')
bathrooms = st.number_input('Enter number of bathrooms')
if st.button('Predict'):
    pred = predict_price(location,sq_ft,bathrooms,bedrooms)
    st.write(f'The price of the house you are looking for is **Rs.{round(pred,2)} lakhs**',font="Helvetica 50 bold")
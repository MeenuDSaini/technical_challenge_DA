import streamlit as st
import pickle
import pandas as pd
from PIL import Image

# Load the model
rf_model = pickle.load(open('rf_model.pkl', 'rb'))

# Set the title with Markdown for font size and color
st.markdown("<h1 style='font-size: 40px; color: blue;'>Fuel Consumption Predictor</h1>", unsafe_allow_html=True)
st.sidebar.header('Condition')
image = Image.open('capture.jpg')
st.image(image, width=800)

# Define the selected features
selected_features = ['distance', 'speed', 'temp_inside', 'temp_outside', 'AC', 'rain', 'sun', 'temp_diff']

min_values = {'distance': 0, 'speed': 0, 'temp_inside': 0, 'temp_outside': -5, 'AC': 0, 'rain': 0, 'sun': 0, 'temp_diff': -8}
max_values = {'distance': 250, 'speed': 100, 'temp_inside': 30, 'temp_outside': 35, 'AC': 1, 'rain': 1, 'sun': 1, 'temp_diff': 30}

# FUNCTION
def user_report():
    user_report_data = {}
    for feature in selected_features:
        user_report_data[feature] = st.sidebar.slider(f'{feature}', min_values[feature], max_values[feature], min_values[feature])

    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data

user_data = user_report()

st.header('User Input')
st.write(user_data)

consume = rf_model.predict(user_data[selected_features].values)
rounded_consume = round(consume[0])
st.header('User Input')
st.write(user_data)
st.subheader('Fuel Consumption')
st.write(f"<div style='font-size: 24px; color: red;'>{rounded_consume}</div>", unsafe_allow_html=True)  # Display the predicted fuel Consumption

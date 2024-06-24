import pandas as pd
import pickle
import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Load the model
model = pickle.load(open('predictor.pkl', 'rb'))

# Streamlit app
st.title("Player Data Prediction")

st.sidebar.header("Player Data Input")

def user_input_features():
    movement_reactions = st.sidebar.number_input("Movement Reactions", min_value=0, max_value=100, value=0)
    potential = st.sidebar.number_input("Potential", min_value=0, max_value=100, value=0)
    age = st.sidebar.number_input("Age", min_value=0, max_value=100, value=0)
    mentality_composure = st.sidebar.number_input("Mentality Composure", min_value=0, max_value=100, value=0)
    value_eur = st.sidebar.number_input("Value (EUR)", min_value=0, max_value=10000000000, value=0)
    attacking_short_passing = st.sidebar.number_input("Attacking Short Passing", min_value=0, max_value=100, value=0)
    mentality_vision = st.sidebar.number_input("Mentality Vision", min_value=0, max_value=100, value=0)
    international_reputation = st.sidebar.number_input("International Reputation", min_value=0, max_value=100, value=0)
    skill_long_passing = st.sidebar.number_input("Skill Long Passing", min_value=0, max_value=100, value=0)
    passing = st.sidebar.number_input("Passing", min_value=0, max_value=100, value=0)
    dribbling = st.sidebar.number_input("Dribbling", min_value=0, max_value=100, value=0)
    physic = st.sidebar.number_input("Physic", min_value=0, max_value=100, value=0)
    wage_eur = st.sidebar.number_input("Wage (EUR)", min_value=0, max_value=100000000, value=0)
    power_shot_power = st.sidebar.number_input("Power Shot Power", min_value=0, max_value=100, value=0)
    skill_ball_control = st.sidebar.number_input("Skill Ball Control", min_value=0, max_value=100, value=0)

    data = {
        'movement_reactions': movement_reactions,
        'potential': potential,
        'age': age,
        'mentality_composure': mentality_composure,
        'value_eur': value_eur,
        'attacking_short_passing': attacking_short_passing,
        'mentality_vision': mentality_vision,
        'international_reputation': international_reputation,
        'skill_long_passing': skill_long_passing,
        'passing': passing,
        'dribbling': dribbling,
        'physic': physic,
        'wage_eur': wage_eur,
        'power_shot_power': power_shot_power,
        'skill_ball_control': skill_ball_control
    }
    return data

input_data = user_input_features()

if st.sidebar.button('Predict'):
    # Convert data to the appropriate format and apply scaling
    features = np.array([[
        input_data['movement_reactions'], input_data['potential'], input_data['age'], input_data['mentality_composure'],
        input_data['value_eur'], input_data['attacking_short_passing'], input_data['mentality_vision'],
        input_data['international_reputation'], input_data['skill_long_passing'], input_data['passing'],
        input_data['dribbling'], input_data['physic'], input_data['wage_eur'],
        input_data['power_shot_power'], input_data['skill_ball_control']
    ]], dtype=float)

    # Make prediction
    prediction = model.predict(features)

    st.subheader("Prediction Result")
    st.write(f"Predicted Value: {prediction[0]}")


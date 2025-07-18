import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load model
pipe = pickle.load(open('pipe.pkl', 'rb'))

st.title("🚢 Titanic Survival Predictor")
st.write("Enter passenger details to predict survival.")

# Input fields
pclass = st.selectbox('Passenger Class (Pclass)', [1, 2, 3])
sex = st.selectbox('Sex', ['male', 'female'])
age = st.number_input('Age', min_value=0.42, max_value=80.0, step=0.1)
sibsp = st.number_input('Siblings/Spouses Aboard (SibSp)', min_value=0, max_value=10)
parch = st.number_input('Parents/Children Aboard (Parch)', min_value=0, max_value=10)
fare = st.number_input('Fare', min_value=0.0, step=0.1)
embarked = st.selectbox('Port of Embarkation', ['S', 'C', 'Q'])

# Predict
if st.button('Predict'):
    input_df = pd.DataFrame([[
        pclass,
        sex,
        age,
        sibsp,
        parch,
        fare,
        embarked
    ]], columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])

    prediction = pipe.predict(input_df)
    result = '✅ Survived 😇' if prediction[0] == 1 else '❌ Did not survive 💀'
    st.success(f"Prediction: {result}")
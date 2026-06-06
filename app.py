import streamlit as st
import pickle
import pandas as pd

# Load model
with openmodel_path = r"C:\Users\DELL\OneDrive\Desktop\ML_Flow\mlruns\2\models\m-37e6199761b844aeb02fd3031865f8e8\artifacts\model.pkl"
    model = pickle.load(f)

st.set_page_config(
    page_title="Iris Flower Prediction",
    page_icon="🌸",
    layout="centered"
)

st.title("🌸 Iris Flower Prediction App")

st.write("Enter flower measurements below:")

# Inputs
sepal_length = st.number_input(
    "Sepal Length (cm)",
    min_value=0.0,
    value=5.1
)

sepal_width = st.number_input(
    "Sepal Width (cm)",
    min_value=0.0,
    value=3.5
)

petal_length = st.number_input(
    "Petal Length (cm)",
    min_value=0.0,
    value=1.4
)

petal_width = st.number_input(
    "Petal Width (cm)",
    min_value=0.0,
    value=0.2
)

if st.button("Predict"):

    input_data = pd.DataFrame({
        "sepal length (cm)": [sepal_length],
        "sepal width (cm)": [sepal_width],
        "petal length (cm)": [petal_length],
        "petal width (cm)": [petal_width]
    })

    prediction = model.predict(input_data)[0]

    classes = {
        0: "Setosa",
        1: "Versicolor",
        2: "Virginica"
    }

    st.success(
        f"Predicted Flower: {classes[prediction]}"
    )
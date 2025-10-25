# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(page_title="Wine Quality Prediction", layout="wide")

st.title("🍷 Wine Quality Prediction App")

st.markdown("""
This app lets you manually enter wine chemical properties,  
applies a saved **PowerTransformer**, and predicts the **Wine Quality**  
using your trained ML model (`wine_quality_model.pkl`).
""")

# -------------------------------------------------
# 1️⃣ Define your feature columns (from dataset)
# -------------------------------------------------
numeric_cols = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide",
    "density", "pH", "sulphates", "alcohol"
]

# (Add categorical if your dataset has any)
categorical_cols = []

# -------------------------------------------------
# 2️⃣ Manual input section
# -------------------------------------------------
st.header("Enter Wine Features")

with st.form("input_form"):
    input_data = {}

    # Continuous variables
    st.subheader("Numeric Features")
    for col in numeric_cols:
        input_data[col] = st.number_input(f"{col}", value=0.0, format="%.5f")

    # If categorical exist, show dropdowns
    if categorical_cols:
        st.subheader("Categorical Features")
        for col in categorical_cols:
            input_data[col] = st.selectbox(f"{col}", options=["Option1", "Option2"])

    submitted = st.form_submit_button("Predict")

# -------------------------------------------------
# 3️⃣ Load model and transformer
# -------------------------------------------------
if submitted:
    st.write("---")
    st.write("### Input Summary")
    row = pd.DataFrame([input_data])
    st.dataframe(row)

    # Load PowerTransformer
    if os.path.exists("power_transformer.pkl"):
        with open("power_transformer.pkl", "rb") as f:
            pt = pickle.load(f)
        st.success("✅ PowerTransformer loaded successfully.")
    else:
        st.error("❌ power_transformer.pkl not found!")
        st.stop()

    # Load ML model
    if os.path.exists("wine_quality_model.pkl"):
        with open("wine_quality_model.pkl", "rb") as f:
            model = pickle.load(f)
        st.success("✅ Wine Quality model loaded successfully.")
    else:
        st.error("❌ wine_quality_model.pkl not found!")
        st.stop()

    # -------------------------------------------------
    # 4️⃣ Apply transformation and predict
    # -------------------------------------------------
    try:
        # Apply PowerTransformer on numeric inputs
        X_num = row[numeric_cols]
        X_transformed = pt.transform(X_num)
        X_final = pd.DataFrame(X_transformed, columns=numeric_cols)

        # Predict
        prediction = model.predict(X_final)

        st.success("🎯 Predicted Wine Quality:")
        st.write(f"**{prediction[0]}**")

    except Exception as e:
        st.error(f"❌ Error during transformation or prediction: {e}")

# -------------------------------------------------
# 5️⃣ Notes
# -------------------------------------------------
st.write("---")
st.caption("""
💡 **Notes:**
- Keep both `wine_quality_model.pkl` and `power_transformer.pkl` in the same directory as `app.py`.
- Make sure the numeric columns entered here match the training dataset used for fitting the transformer and model.
""")

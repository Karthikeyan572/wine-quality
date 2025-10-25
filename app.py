# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(page_title="ML Model Prediction App", layout="wide")

st.title("🍷 Machine Learning Prediction App with PowerTransformer + Model")

st.markdown("""
This app allows manual input of **continuous** and **categorical** variables,
applies preprocessing using a saved **PowerTransformer (pt.pkl)**,
and then predicts using your **trained model (dt.pkl)**.
""")

# ----------------------------
# 1️⃣  Define schema manually
# ----------------------------

# 👇 Change these to match your dataset
numeric_cols = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide",
    "density", "pH", "sulphates", "alcohol"
]
categorical_cols = []  # if any categorical features exist, list them here

# ----------------------------
# 2️⃣  Create manual input form
# ----------------------------
st.header("Enter input values manually")

with st.form("input_form"):
    input_data = {}

    # Continuous inputs
    st.subheader("Continuous (numeric) variables")
    for col in numeric_cols:
        input_data[col] = st.number_input(f"{col}", value=0.0, format="%.5f")

    # Categorical inputs (dropdowns)
    if categorical_cols:
        st.subheader("Categorical variables")
        for col in categorical_cols:
            input_data[col] = st.selectbox(f"{col}", options=["Option1", "Option2"])

    submitted = st.form_submit_button("Predict")

# ----------------------------
# 3️⃣  On submit — predict
# ----------------------------
if submitted:
    st.write("---")
    st.write("### Input Summary")
    row = pd.DataFrame([input_data])
    st.dataframe(row)

    # Load PowerTransformer (pt.pkl)
    if os.path.exists("pt.pkl"):
        try:
            with open("pt.pkl", "rb") as f:
                pt = pickle.load(f)
            st.success("✅ PowerTransformer (pt.pkl) loaded successfully.")
        except Exception as e:
            st.error(f"❌ Error loading pt.pkl: {e}")
            st.stop()
    else:
        st.warning("⚠️ No pt.pkl found. Proceeding without transformation.")
        pt = None

    # Load Model (dt.pkl)
    if os.path.exists("dt.pkl"):
        try:
            with open("dt.pkl", "rb") as f:
                model = pickle.load(f)
            st.success("✅ Model (dt.pkl) loaded successfully.")
        except Exception as e:
            st.error(f"❌ Error loading dt.pkl: {e}")
            st.stop()
    else:
        st.error("❌ No dt.pkl found in current directory.")
        st.stop()

    # Preprocess numeric data using PowerTransformer
    X_num = row[numeric_cols].copy()

    if pt is not None:
        try:
            X_num_transformed = pt.transform(X_num)
            st.info("Numeric values transformed using PowerTransformer.")
        except Exception as e:
            st.error(f"Error applying PowerTransformer: {e}")
            X_num_transformed = X_num.values
    else:
        X_num_transformed = X_num.values

    # Recombine (in case you have categorical columns)
    X_final = pd.DataFrame(X_num_transformed, columns=numeric_cols)

    # Predict
    try:
        prediction = model.predict(X_final)
        st.success("🎯 Prediction Result:")
        st.write(prediction)
    except Exception as e:
        st.error(f"❌ Error during model prediction: {e}")

# ----------------------------
# 4️⃣  Notes
# ----------------------------
st.write("---")
st.caption("""
💡 **Tips:**
- Ensure both `dt.pkl` and `pt.pkl` are in the same directory.
- `pt.pkl` should be fitted on your training data’s numeric columns.
- If your model expects encoded categorical variables, include that encoder too or integrate it in a pipeline before dumping.
""")

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import PowerTransformer

# Set page configuration
st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #8B0000;
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🍷 Wine Quality Prediction App")
st.markdown("### Predict wine quality: Bad, Better, or Best")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This app predicts wine quality based on physicochemical properties.
    
    **Model:** Decision Tree Classifier
    **Classes:** Bad, Better, Best
    **Accuracy:** ~99%
    """)
    
    st.markdown("---")
    st.caption("Built with Streamlit 🎈")

# Load model
@st.cache_resource
def load_model():
    try:
        if os.path.exists('wine_quality_model.pkl'):
            with open('wine_quality_model.pkl', 'rb') as f:
                model = pickle.load(f)
            return model, True
        else:
            st.warning("⚠️ Model file 'wine_quality_model.pkl' not found. Please upload it.")
            return None, False
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, False

model, model_loaded = load_model()

# Input form
st.subheader("🔬 Enter Wine Properties")

col1, col2, col3 = st.columns(3)

with col1:
    fixed_acidity = st.number_input(
        "Fixed Acidity",
        min_value=4.0,
        max_value=16.0,
        value=7.4,
        step=0.1,
        help="Most wines: 6.0 - 10.0"
    )
    
    volatile_acidity = st.number_input(
        "Volatile Acidity",
        min_value=0.1,
        max_value=2.0,
        value=0.5,
        step=0.01,
        help="Most wines: 0.3 - 0.7"
    )
    
    citric_acid = st.number_input(
        "Citric Acid",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.01,
        help="Most wines: 0.0 - 0.5"
    )
    
    residual_sugar = st.number_input(
        "Residual Sugar",
        min_value=0.5,
        max_value=16.0,
        value=2.3,
        step=0.1,
        help="Most wines: 1.5 - 3.0"
    )

with col2:
    chlorides = st.number_input(
        "Chlorides",
        min_value=0.01,
        max_value=0.7,
        value=0.08,
        step=0.01,
        help="Most wines: 0.05 - 0.10"
    )
    
    free_sulfur_dioxide = st.number_input(
        "Free Sulfur Dioxide",
        min_value=1.0,
        max_value=80.0,
        value=15.0,
        step=1.0,
        help="Most wines: 5 - 25"
    )
    
    total_sulfur_dioxide = st.number_input(
        "Total Sulfur Dioxide",
        min_value=6.0,
        max_value=300.0,
        value=46.0,
        step=1.0,
        help="Most wines: 20 - 80"
    )
    
    density = st.number_input(
        "Density",
        min_value=0.990,
        max_value=1.005,
        value=0.9967,
        step=0.0001,
        format="%.4f",
        help="Most wines: 0.994 - 0.998"
    )

with col3:
    pH = st.number_input(
        "pH",
        min_value=2.5,
        max_value=4.5,
        value=3.3,
        step=0.01,
        help="Most wines: 3.0 - 3.5"
    )
    
    sulphates = st.number_input(
        "Sulphates",
        min_value=0.3,
        max_value=2.5,
        value=0.65,
        step=0.01,
        help="Most wines: 0.5 - 0.8"
    )
    
    alcohol = st.number_input(
        "Alcohol (%)",
        min_value=8.0,
        max_value=15.0,
        value=10.4,
        step=0.1,
        help="Most wines: 9.0 - 12.0"
    )

st.markdown("---")

# Predict button
if st.button("🔮 Predict Wine Quality", type="primary"):
    if not model_loaded:
        st.error("❌ Cannot make prediction. Model not loaded.")
    else:
        with st.spinner("Analyzing wine properties..."):
            # Create input dataframe
            input_data = pd.DataFrame({
                'fixed acidity': [fixed_acidity],
                'volatile acidity': [volatile_acidity],
                'citric acid': [citric_acid],
                'residual sugar': [residual_sugar],
                'chlorides': [chlorides],
                'free sulfur dioxide': [free_sulfur_dioxide],
                'total sulfur dioxide': [total_sulfur_dioxide],
                'density': [density],
                'pH': [pH],
                'sulphates': [sulphates],
                'alcohol': [alcohol]
            })
            
            # Apply PowerTransformer (same as training)
            cols_to_transform = ['fixed acidity', 'volatile acidity', 'residual sugar', 
                                'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 
                                'sulphates', 'alcohol']
            
            pt = PowerTransformer(method="yeo-johnson")
            input_data[cols_to_transform] = pt.fit_transform(input_data[cols_to_transform])
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            
            # Get prediction probabilities if available
            try:
                prediction_proba = model.predict_proba(input_data)[0]
                class_names = model.classes_
                proba_dict = dict(zip(class_names, prediction_proba))
            except:
                proba_dict = None
            
            # Display results
            st.success("✅ Prediction Complete!")
            
            st.markdown("---")
            st.subheader("🎯 Prediction Results")
            
            # Color code results
            if prediction == "best":
                st.success(f"### Quality: {prediction.upper()} 🌟🌟🌟")
                color = "#28a745"
            elif prediction == "better":
                st.info(f"### Quality: {prediction.upper()} ⭐⭐")
                color = "#17a2b8"
            else:
                st.warning(f"### Quality: {prediction.upper()} ⭐")
                color = "#ffc107"
            
            # Display probabilities
            if proba_dict:
                st.markdown("---")
                st.subheader("📊 Confidence Scores")
                
                col_prob1, col_prob2, col_prob3 = st.columns(3)
                
                with col_prob1:
                    st.metric(
                        "Bad", 
                        f"{proba_dict.get('bad', 0)*100:.1f}%",
                        delta=None
                    )
                
                with col_prob2:
                    st.metric(
                        "Better", 
                        f"{proba_dict.get('better', 0)*100:.1f}%",
                        delta=None
                    )
                
                with col_prob3:
                    st.metric(
                        "Best", 
                        f"{proba_dict.get('best', 0)*100:.1f}%",
                        delta=None
                    )
            
            # Display input summary
            st.markdown("---")
            st.subheader("📋 Input Summary")
            
            input_summary = pd.DataFrame({
                'Property': [
                    'Fixed Acidity', 'Volatile Acidity', 'Citric Acid', 
                    'Residual Sugar', 'Chlorides', 'Free SO₂', 
                    'Total SO₂', 'Density', 'pH', 'Sulphates', 'Alcohol'
                ],
                'Value': [
                    fixed_acidity, volatile_acidity, citric_acid,
                    residual_sugar, chlorides, free_sulfur_dioxide,
                    total_sulfur_dioxide, density, pH, sulphates, alcohol
                ]
            })
            
            st.dataframe(input_summary, use_container_width=True, hide_index=True)

# Instructions for model upload
if not model_loaded:
    with st.expander("📝 How to add your trained model"):
        st.markdown("""
        ### Steps to use your model:
        
        1. **Ensure your model is saved:**
        ```
        import pickle
        
        # After training (from your code)
        with open('wine_quality_model.pkl', 'wb') as f:
            pickle.dump(dt, f)
        ```
        
        2. **Upload the model file:**
        - Place `wine_quality_model.pkl` in the same directory as `app.py`
        
        3. **Restart the app:**
        - The model will be automatically loaded
        
        ### Your training code uses:
        - PowerTransformer (yeo-johnson) on specific columns
        - Decision Tree Classifier
        - 3 classes: bad, better, best
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Wine Quality Prediction | Trained on Red Wine Dataset</p>
</div>
""", unsafe_allow_html=True)

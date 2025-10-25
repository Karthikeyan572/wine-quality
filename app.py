import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

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

# Load model and PowerTransformer at the start
@st.cache_resource
def load_pickle_files():
    """Load both the model and PowerTransformer pickle files"""
    model = None
    power_transformer = None
    model_loaded = False
    pt_loaded = False
    error_messages = []
    
    try:
        # Load the Decision Tree model
        with open('wine_quality_model.pkl', 'rb') as f:
            model = pickle.load(f)
        model_loaded = True
    except FileNotFoundError:
        error_messages.append("❌ 'wine_quality_model.pkl' not found")
    except Exception as e:
        error_messages.append(f"❌ Error loading model: {str(e)}")
    
    try:
        # Load the PowerTransformer
        with open('power_transformer.pkl', 'rb') as f:
            power_transformer = pickle.load(f)
        pt_loaded = True
    except FileNotFoundError:
        error_messages.append("❌ 'power_transformer.pkl' not found")
    except Exception as e:
        error_messages.append(f"❌ Error loading PowerTransformer: {str(e)}")
    
    return model, power_transformer, model_loaded, pt_loaded, error_messages

# Load the pickle files
model, power_transformer, model_loaded, pt_loaded, error_messages = load_pickle_files()

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
    
    # Show file loading status
    st.subheader("📦 Pickle Files Status")
    
    if model_loaded:
        st.success("✅ Model loaded successfully!")
    else:
        st.error("❌ Model not loaded")
    
    if pt_loaded:
        st.success("✅ PowerTransformer loaded successfully!")
    else:
        st.error("❌ PowerTransformer not loaded")
    
    # Display error messages if any
    if error_messages:
        st.markdown("---")
        st.subheader("⚠️ Errors:")
        for msg in error_messages:
            st.write(msg)
    
    st.markdown("---")
    st.caption("Built with Streamlit 🎈")

# Show warning if files not loaded
if not (model_loaded and pt_loaded):
    st.error("⚠️ Required pickle files are missing! Please upload them to continue.")
    st.info("""
    **Required files:**
    1. `wine_quality_model.pkl` - Your trained Decision Tree model
    2. `power_transformer.pkl` - Your fitted PowerTransformer
    
    Place both files in the same directory as app.py
    """)

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
    if not (model_loaded and pt_loaded):
        st.error("❌ Cannot make prediction. Required pickle files are not loaded.")
    else:
        with st.spinner("Analyzing wine properties..."):
            try:
                # Create input dataframe with exact column names from training
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
                
                # Apply PowerTransformer on the same columns as training
                cols_to_transform = ['fixed acidity', 'volatile acidity', 'residual sugar', 
                                    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 
                                    'sulphates', 'alcohol']
                
                # Transform using the loaded PowerTransformer
                input_data[cols_to_transform] = power_transformer.transform(input_data[cols_to_transform])
                
                # Make prediction using the loaded model
                prediction = model.predict(input_data)[0]
                
                # Get prediction probabilities
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
                
                # Color code results based on prediction
                if prediction == "best":
                    st.success(f"### Quality: {prediction.upper()} 🌟🌟🌟")
                elif prediction == "better":
                    st.info(f"### Quality: {prediction.upper()} ⭐⭐")
                else:
                    st.warning(f"### Quality: {prediction.upper()} ⭐")
                
                # Display confidence scores
                if proba_dict:
                    st.markdown("---")
                    st.subheader("📊 Confidence Scores")
                    
                    col_prob1, col_prob2, col_prob3 = st.columns(3)
                    
                    with col_prob1:
                        st.metric(
                            "Bad", 
                            f"{proba_dict.get('bad', 0)*100:.1f}%"
                        )
                    
                    with col_prob2:
                        st.metric(
                            "Better", 
                            f"{proba_dict.get('better', 0)*100:.1f}%"
                        )
                    
                    with col_prob3:
                        st.metric(
                            "Best", 
                            f"{proba_dict.get('best', 0)*100:.1f}%"
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
            
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")
                st.write("Debug info:", str(e))

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Wine Quality Prediction | Trained on Red Wine Dataset</p>
</div>
""", unsafe_allow_html=True)

import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'wine_quality_model.pkl'
CSV_PATH = BASE_DIR / 'winequality-red.csv'


def load_model(path=MODEL_PATH):
    if not path.exists():
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


@st.cache_data
def load_csv(path=CSV_PATH):
    if path.exists():
        return pd.read_csv(path)
    return None


def get_feature_names(df):
    if df is None:
        # Fallback list (common wine dataset columns)
        return [
            'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
            'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
            'pH', 'sulphates', 'alcohol'
        ]
    return [c for c in df.columns if c not in ('quality', 'quality_label')]


def main():
    st.title('Wine Quality Predictor (Streamlit)')

    st.markdown('Enter the wine features below and click Predict. The app will load `wine_quality_model.pkl` from the same folder.')

    df = load_csv()
    feature_names = get_feature_names(df)

    model = load_model()
    if model is None:
        st.warning('Model not found: place `wine_quality_model.pkl` in the project folder to enable predictions.')

    # Prepare default values (means when CSV available)
    defaults = {}
    if df is not None:
        for f in feature_names:
            if f in df.columns:
                defaults[f] = float(df[f].mean())
            else:
                defaults[f] = 0.0
    else:
        for f in feature_names:
            defaults[f] = 0.0

    st.subheader('Input features')
    cols = st.columns(3)
    inputs = {}
    for i, feat in enumerate(feature_names):
        col = cols[i % 3]
        val = col.number_input(feat, value=defaults.get(feat, 0.0), format="%.6f")
        inputs[feat] = float(val)

    if st.button('Predict'):
        X = np.array([inputs[f] for f in feature_names]).reshape(1, -1)
        if model is None:
            st.error('Model not loaded. Cannot predict.')
        else:
            try:
                pred = model.predict(X)
                label = pred[0]
                # If label is numeric, show as int; otherwise show str
                try:
                    disp_label = int(label)
                except Exception:
                    disp_label = str(label)

                st.success(f'Predicted quality label: {disp_label}')

                # If model supports probabilities, show them with class names
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)[0]
                    # Attempt to extract class names if available
                    classes = None
                    if hasattr(model, 'classes_'):
                        classes = list(model.classes_)
                    # Build a labeled dict
                    prob_dict = {}
                    for i, p in enumerate(proba):
                        key = classes[i] if classes is not None else str(i)
                        prob_dict[str(key)] = float(p)
                    st.write('Class probabilities:')
                    st.write(prob_dict)
            except Exception as e:
                st.error(f'Error during prediction: {e}')


if __name__ == '__main__':
    main()

import os
import pickle
from flask import Flask, render_template, request, redirect, url_for, flash
import numpy as np

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Path to the pickled model (created in the notebook)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'wine_quality_model.pkl')

# Features expected by the model in the same order used during training
FEATURE_NAMES = [
    'fixed acidity', 'volatile acidity', 'residual sugar', 'chlorides',
    'free sulfur dioxide', 'total sulfur dioxide', 'sulphates', 'alcohol',
    'citric acid', 'density', 'pH'
]


def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}. Make sure 'wine_quality_model.pkl' exists.")
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model


model = None
try:
    model = load_model()
except Exception as e:
    # We'll still start the app; errors will be shown on the page if model missing
    print(f"Warning: could not load model: {e}")


@app.route('/', methods=['GET'])
def index():
    # Render a form with all feature names
    return render_template('index.html', feature_names=FEATURE_NAMES, result=None)


@app.route('/predict', methods=['POST'])
def predict():
    global model
    if model is None:
        try:
            model = load_model()
        except Exception as e:
            flash(str(e), 'danger')
            return redirect(url_for('index'))

    # Collect inputs
    values = []
    errors = []
    for name in FEATURE_NAMES:
        v = request.form.get(name)
        if v is None or v.strip() == '':
            errors.append(f"Missing value for '{name}'")
            values.append(0)
            continue
        try:
            fv = float(v)
        except ValueError:
            errors.append(f"Invalid numeric value for '{name}': {v}")
            fv = 0.0
        values.append(fv)

    if errors:
        for e in errors:
            flash(e, 'warning')

    # Model expects 2D array
    X = np.array(values).reshape(1, -1)
    try:
        pred = model.predict(X)
        proba = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)
        result = {
            'prediction': int(pred[0]),
            'probability': proba.tolist() if proba is not None else None,
            'inputs': dict(zip(FEATURE_NAMES, values))
        }
    except Exception as e:
        flash(f"Error during prediction: {e}", 'danger')
        return redirect(url_for('index'))

    return render_template('index.html', feature_names=FEATURE_NAMES, result=result)


if __name__ == '__main__':
    app.run(debug=True)

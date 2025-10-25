from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer

app = Flask(__name__)

# Load the trained model and transformer
try:
    with open('wine_quality_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('power_transformer.pkl', 'rb') as f:
        pt = pickle.load(f)
    print("Model and transformer loaded successfully!")
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    print("Please ensure 'wine_quality_model.pkl' and 'power_transformer.pkl' exist in the same directory.")

# Feature names that need transformation (same as in training)
transform_cols = ['fixed acidity', 'volatile acidity', 'residual sugar', 'chlorides', 
                  'free sulfur dioxide', 'total sulfur dioxide', 'sulphates', 'alcohol']

# All feature names in order
all_features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 
                'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 
                'pH', 'sulphates', 'alcohol']

@app.route('/')
def home():
    """Render the home page with the prediction form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle form submission and return prediction"""
    try:
        # Get data from form
        features = {
            'fixed acidity': float(request.form['fixed_acidity']),
            'volatile acidity': float(request.form['volatile_acidity']),
            'citric acid': float(request.form['citric_acid']),
            'residual sugar': float(request.form['residual_sugar']),
            'chlorides': float(request.form['chlorides']),
            'free sulfur dioxide': float(request.form['free_sulfur_dioxide']),
            'total sulfur dioxide': float(request.form['total_sulfur_dioxide']),
            'density': float(request.form['density']),
            'pH': float(request.form['pH']),
            'sulphates': float(request.form['sulphates']),
            'alcohol': float(request.form['alcohol'])
        }
        
        # Create DataFrame with features in correct order
        input_df = pd.DataFrame([features])
        
        # Apply power transformation to the same columns as training
        input_df[transform_cols] = pt.transform(input_df[transform_cols])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]
        
        # Get class probabilities
        classes = model.classes_
        probabilities = {cls: round(prob * 100, 2) for cls, prob in zip(classes, prediction_proba)}
        
        # Get confidence level
        max_prob = max(probabilities.values())
        confidence = "High" if max_prob > 80 else "Medium" if max_prob > 60 else "Low"
        
        return render_template('index.html', 
                             prediction=prediction,
                             probabilities=probabilities,
                             confidence=confidence,
                             input_features=features)
    
    except ValueError as ve:
        return render_template('index.html', error=f"Invalid input: {str(ve)}")
    except Exception as e:
        return render_template('index.html', error=f"Prediction error: {str(e)}")

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for JSON-based predictions"""
    try:
        # Get JSON data
        data = request.get_json()
        
        # Validate that all required features are present
        missing_features = [f for f in all_features if f not in data]
        if missing_features:
            return jsonify({
                'error': f'Missing required features: {", ".join(missing_features)}'
            }), 400
        
        # Create DataFrame
        input_df = pd.DataFrame([data])
        
        # Apply power transformation
        input_df[transform_cols] = pt.transform(input_df[transform_cols])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]
        
        # Get class probabilities
        classes = model.classes_
        probabilities = {cls: float(prob) for cls, prob in zip(classes, prediction_proba)}
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'probabilities': probabilities,
            'confidence': max(probabilities.values())
        })
    
    except ValueError as ve:
        return jsonify({'error': f'Invalid input values: {str(ve)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/about')
def about():
    """Information about the model"""
    model_info = {
        'model_type': 'Decision Tree Classifier',
        'features': all_features,
        'classes': ['bad', 'better', 'best'],
        'description': 'This model predicts wine quality based on physicochemical properties.',
        'preprocessing': 'Yeo-Johnson Power Transformation applied to selected features'
    }
    return jsonify(model_info)

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return render_template('index.html', error='Page not found'), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors"""
    return render_template('index.html', error='Internal server error'), 500

if __name__ == '__main__':
    print("Starting Wine Quality Prediction Server...")
    print("Access the application at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)

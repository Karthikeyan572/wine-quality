# Wine Quality Predictor (Flask)

This simple Flask app loads the trained model `wine_quality_model.pkl` (created in the notebook) and provides a web form to enter feature values and predict the wine quality label (1 for good quality, 0 for not good).

Files added:
- `app.py` - Flask application
- `templates/index.html` - HTML form + result display
- `requirments.txt` - Python dependencies (note filename is misspelled as in the repo)

Run (Windows PowerShell):

```powershell
# Activate your virtual environment if needed, then:
pip install -r .\requirments.txt
python .\app.py
```

Open http://127.0.0.1:5000/ in your browser. Ensure `wine_quality_model.pkl` is in the same folder as `app.py`.

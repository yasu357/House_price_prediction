from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = load_model('house_price_model.h5')
scaler = joblib.load('standard_scaler.pkl')

# Define the main 5 features
features = ['sqft_living', 'sqft_lot', 'bedrooms', 'bathrooms', 'condition']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form

    # Preprocess features
    new_data = [float(data[feature]) for feature in features]
    new_data_scaled = scaler.transform([new_data])

    # Make predictions using the trained model
    prediction = model.predict(new_data_scaled)[0][0]

    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(port=5000, debug=True)

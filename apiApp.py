from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import json

app = Flask(__name__)

# Load the pre-trained model
model = load_model('my_model.h5')

# Load the model history (metrics)
history = pd.read_csv('history.csv')

# Load the preprocessor (standard scaler, encoders, etc.) used for scaling data
# You should save the preprocessor after training and load it here
# In this case, it's recreated, but ideally, you'd persist it
categorical_columns = ['Scholarship type', 'Regular artistic or sports activity']
numerical_columns = ['Student Age', 'Additional work', 'Total salary if available',
                     'Weekly study hours', 'Attendance to classes',
                     'Preparation to midterm exams 1',
                     'Cumulative grade point average in the last semester (/4.00)']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(), categorical_columns)
    ])

# For simplicity, fit the preprocessor with a dummy DataFrame; replace with your actual training data
dummy_data = pd.DataFrame(np.random.rand(10, len(numerical_columns)), columns=numerical_columns)
dummy_data[categorical_columns] = np.random.choice(['Full', 'Partial'], size=(10, len(categorical_columns)))
preprocessor.fit(dummy_data)


# -------------------- API Routes --------------------
# Define the root route
@app.route('/')
def home():
    return "Welcome to the Student Performance Prediction API!"

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Expect input as JSON

    # Extract features from the request
    features = [
        data['Student Age'],
        data['Additional work'],
        data['Total salary if available'],
        data['Weekly study hours'],
        data['Attendance to classes'],
        data['Preparation to midterm exams 1'],
        data['Cumulative grade point average in the last semester (/4.00)'],
        data['Scholarship type'],
        data['Regular artistic or sports activity']
    ]

    # Convert input to DataFrame
    input_data = pd.DataFrame([features], columns=numerical_columns + categorical_columns)

    # Preprocess the input
    input_scaled = preprocessor.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)

    # Return the prediction as JSON
    return jsonify({'prediction': prediction[0].tolist()})


# Metrics route
@app.route('/metrics', methods=['GET'])
def get_metrics():
    # Send model performance metrics (like accuracy and loss) as JSON
    metrics = {
        'epochs': history['epochs'].tolist(),
        'accuracy': history['accuracy'].tolist(),
        'val_accuracy': history['val_accuracy'].tolist(),
        'loss': history['loss'].tolist(),
        'val_loss': history['val_loss'].tolist()
    }

    return jsonify(metrics)


# Feature Importance Route (Optional)
# Add only if you calculate and want to return feature importance
# from a library like SHAP

if __name__ == '__main__':
    app.run(debug=True)

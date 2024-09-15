from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the model
model = load_model('my_model.h5')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = [data['features']]
    prediction = model.predict(features)
    return jsonify({'prediction': prediction[0]})


if __name__ == '__main__':
    app.run(debug=True)

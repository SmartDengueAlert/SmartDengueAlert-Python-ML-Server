from flask import Flask, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('random_forest_model2.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    if not data or 'features' not in data:
        return jsonify({'error': 'Missing required data'}), 400
    
    features = np.array(data['features']).reshape(1, -1)
    
    # Ensure the features array has the correct number of inputs
    if features.shape[1] != 58:
        return jsonify({'error': 'Incorrect number of features'}), 400
    
    # Make prediction
    prediction = model.predict(features)

    # Print the prediction to the console
    print(f"********Dengue prediction: {prediction[0]}")
    
    # Return the prediction as JSON
    return jsonify({'Dengue Prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)


    
  


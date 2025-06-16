#API for Android app (Crop prediction)
import os
from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS

app = Flask(__name__, static_url_path='/images', static_folder='images')
CORS(app)

# Load your trained model
model = joblib.load("model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Extract input features from the JSON
    N = data.get("N")
    P = data.get("P")
    K = data.get("K")
    temperature = data.get("temperature")
    humidity = data.get("humidity")
    ph = data.get("ph")
    rainfall = data.get("rainfall")

    # Make prediction using model
    input_features = [[N, P, K, temperature, humidity, ph, rainfall]]
    prediction = model.predict(input_features)
    crop = prediction[0]

    #image url 
    image_url = f"https://cropapi-production.up.railway.app/images/{crop.lower().replace(' ', '_')}.jpg"

    return jsonify({
        "prediction": crop,
        "image_url": image_url

    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "UP"}), 200

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 for local
    app.run(host='0.0.0.0', port=port)
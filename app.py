from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/', methods=['GET'])
def index():
    return "ML Model API is running"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    prediction = model.predict([data['features']])
    return jsonify({'prediction': prediction.tolist()})

# Expose the app for Vercel
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

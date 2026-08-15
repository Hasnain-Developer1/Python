from flask import Flask, request, jsonify
import joblib
import pandas as pd

model = joblib.load("bmw_price_model.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    return "BMW Price Prediction API"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    input_df = pd.DataFrame([data])
    prediction = model.predict(input_df)[0]

    return jsonify({
        "predicted_price": float(prediction)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
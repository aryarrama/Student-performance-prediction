from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

# Mengambil lokasi folder tempat file Python berada
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Memuat model dari folder yang sama dengan file Python
model_path = os.path.join(BASE_DIR, "gbr_model.joblib")
joblib_model = joblib.load(model_path)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["data"]
    prediction = joblib_model.predict(data)

    return jsonify({"prediction": prediction.tolist()})


if __name__ == "__main__":
    app.run(debug=True)

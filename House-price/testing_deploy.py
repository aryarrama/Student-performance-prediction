from flask import Flask, request, jsonify
import joblib

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Memuat model yang sudah disimpan
joblib_model = joblib.load("gbr_model.joblib")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["data"]

    # Melakukan prediksi
    prediction = joblib_model.predict(data)

    return jsonify({"prediction": prediction.tolist()})


if __name__ == "__main__":
    app.run(debug=True)

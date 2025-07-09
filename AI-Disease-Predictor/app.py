from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model and symptoms
model = joblib.load("disease_model.pkl")
symptoms = joblib.load("symptoms_list.pkl")

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    user_symptoms = request.json['symptoms']
    input_vector = [1 if symptom in user_symptoms else 0 for symptom in symptoms]
    prediction = model.predict([input_vector])[0]
    return jsonify({'disease': prediction})

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

# Load saved files
model = pickle.load(open('model.pkl', 'rb'))        # your trained model
scaler = pickle.load(open('scaler.pkl', 'rb'))      # scaler
columns = pickle.load(open('columns.pkl', 'rb'))    # training columns

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # =========================
        # 1. GET INPUT FROM FORM
        # =========================
        input_dict = {
    'gender': request.form.get('gender'),
    'SeniorCitizen': request.form.get('SeniorCitizen'),
    'Partner': request.form.get('Partner'),
    'Dependents': request.form.get('Dependents'),
    'tenure': request.form.get('tenure'),

    'PhoneService': request.form.get('PhoneService'),
    'MultipleLines': request.form.get('MultipleLines'),

    'InternetService': request.form.get('InternetService'),
    'OnlineSecurity': request.form.get('OnlineSecurity', 'No'),
    'OnlineBackup': request.form.get('OnlineBackup', 'No'),
    'DeviceProtection': request.form.get('DeviceProtection', 'No'),
    'TechSupport': request.form.get('TechSupport', 'No'),
    'StreamingTV': request.form.get('StreamingTV', 'No'),
    'StreamingMovies': request.form.get('StreamingMovies', 'No'),

    'Contract': request.form.get('Contract'),
    'PaperlessBilling': request.form.get('PaperlessBilling', 'Yes'),
    'PaymentMethod': request.form.get('PaymentMethod'),

    'MonthlyCharges': request.form.get('MonthlyCharges'),
    'TotalCharges': request.form.get('TotalCharges')
    }
        # =========================
        # 2. CONVERT TO DATAFRAME
        # =========================
        input_df = pd.DataFrame([input_dict])

        # Convert numeric fields
        input_df['tenure'] = pd.to_numeric(input_df['tenure'])
        input_df['MonthlyCharges'] = pd.to_numeric(input_df['MonthlyCharges'])
        input_df['TotalCharges'] = pd.to_numeric(input_df['TotalCharges'])
        input_df['SeniorCitizen'] = pd.to_numeric(input_df['SeniorCitizen'])

        # =========================
        # 3. APPLY SAME ENCODING
        # =========================
        input_encoded = pd.get_dummies(input_df)

        # Match training columns
        input_encoded = input_encoded.reindex(columns=columns, fill_value=0)

        # =========================
        # 4. SCALE INPUT
        # =========================
        input_scaled = scaler.transform(input_encoded)

        # =========================
        # 5. PREDICT
        # =========================
        prob = model.predict_proba(input_scaled)[:, 1]
        pred = (prob > 0.45).astype(int)

        # =========================
        # 6. OUTPUT
        # =========================
        result = "Customer will Churn" if pred[0] == 1 else "Customer will not Churn"
        confidence = round(prob[0] * 100, 2)

        return render_template('index.html',
                               prediction_text=f'Result: {result}',
                               confidence =confidence)

    except Exception as e:
        return render_template('index.html',
                               prediction_text=f'Error: {str(e)}')


if __name__ == "__main__":
    app.run(debug=True)
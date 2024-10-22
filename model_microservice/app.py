import joblib
#import torch
#import torch.nn as nn
import pandas as pd
import logging
from flask import Flask, request, jsonify

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

app = Flask(__name__)


@app.route('/saludo', methods=['POST'])
def saludo():
    data = request.get_json()
    nombre = data.get('nombre', 'Mundo')

    data = {
        "person_age": 70,
        "person_income": 98000,
        "person_home_ownership": "OWN",
        "person_emp_length": 9.0,
        "loan_intent": "EDUCATION",
        "loan_grade": "A",
        "loan_amnt": 5000,
        "loan_int_rate": 7.9,
        "loan_percent_income": 0.57,
        "cb_person_default_on_file": "Y",
        "cb_person_cred_hist_length": 3
    }
    
    logging.info('step0001')

    preprocessor = joblib.load('preprocessor.pkl')
    
    logging.info('step0002')

    # Crear un DataFrame con los datos
    data = pd.DataFrame(data, index=[0])
    
    logging.info('step0003')

    # Transformar los datos
    preprocessed_data = preprocessor.transform(data)
    
    logging.info('step0004')

    return jsonify({'mensaje': f'Hola, {nombre}!'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

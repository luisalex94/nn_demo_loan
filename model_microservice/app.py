import joblib
import torch
import torch.nn as nn
import pandas as pd
import logging
from flask import Flask, request, jsonify

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

app = Flask(__name__)


# Define the neural network
class LoanApprovalNN(nn.Module):
    def __init__(self):
        super(LoanApprovalNN, self).__init__()              # Call the constructor of the parent class
        self.fc1 = nn.Linear(26, 64)   # Define the first fully connected layer
        self.fc2 = nn.Linear(64, 64)                        # Define the second fully connected layer
        self.fc3 = nn.Linear(64, 64)                        # Define the third fully connected layer
        self.fc4 = nn.Linear(64, 32)                        # Define the fourth fully connected layer
        self.output = nn.Linear(32, 1)                      # Define the output layer
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))             # Apply the ReLU activation function to the first fully connected layer
        x = torch.relu(self.fc2(x))             # Apply the ReLU activation function to the second fully connected layer
        x = torch.relu(self.fc3(x))             # Apply the ReLU activation function to the third fully connected layer
        x = torch.relu(self.fc4(x))             # Apply the ReLU activation function to the fourth fully connected layer  
        x = torch.sigmoid(self.output(x))       # Apply the sigmoid activation function to the output layer
        return x

# Instantiate the model
model = LoanApprovalNN()

# Load the model
model = torch.load('model.pth')

# Set the model to evaluation mode
# model.eval()


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

import json
import joblib
import torch
import torch.nn as nn
import pandas as pd

# Define the neural network
class LoanApprovalNN(nn.Module):
    def __init__(self):
        super(LoanApprovalNN, self).__init__()              # Call the constructor of the parent class
        self.fc1 = nn.Linear(26, 64)                        # Define the first fully connected layer
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
model.eval()


def lambda_handler(event, context):
    # Example of processing input event
    data = {
        "person_age": 22,
        "person_income": 59000,
        "person_home_ownership": "RENT",
        "person_emp_length": 12.0,
        "loan_intent": "PERSONAL",
        "loan_grade": "D",
        "loan_amnt": 35000,
        "loan_int_rate": 16.02,
        "loan_percent_income": 0.59,
        "cb_person_default_on_file": "Y",
        "cb_person_cred_hist_length": 3
    }
    
    # Load the preprocessor
    preprocessor = joblib.load("preprocessor.pkl")

    # Crear un DataFrame con los datos
    data = pd.DataFrame(data, index=[0])

    # Transformar los datos
    preprocessed_data = preprocessor.transform(data)

    # Example of processing input event
    name = event.get('name', 'World')

    # Convertir los datos preprocesados a tensor y ajustar la forma
    X_tensor = torch.tensor(preprocessed_data, dtype=torch.float32)

    # Realizar las predicciones
    with torch.no_grad():
        predictions = model(X_tensor)

    # Aplicar sigmoide si es necesario
    predictions = torch.sigmoid(predictions)

    # Imprimir resultados
    print(predictions)
    
    # Example of creating a response
    response = {
        'statusCode': 200,
        'body': json.dumps(f'Hello, {name}!')
    }
    
    return response


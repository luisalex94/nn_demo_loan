print('Counter 00001')
print('INIT - KonfioTesting 000')

import json
import joblib
import torch
import torch.nn as nn
import pandas as pd

print('INIT - KonfioTesting 001')

def lambda_handler(event, context):
    # TODO implement
    print('INIT - KonfioTesting 002')
    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }

import json

def lambda_handler(event, context):
    # Example of processing input event
    name = event.get('name', 'World')
    
    # Example of creating a response
    response = {
        'statusCode': 200,
        'body': json.dumps(f'Hello, {name}!')
    }
    
    return response
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/saludo', methods=['POST'])
def saludo():
    data = request.get_json()
    nombre = data.get('nombre', 'Mundo')
    return jsonify({'mensaje': f'Hola, {nombre}!'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
from flask import Flask, request, render_template, jsonify
from generate_response import generate_answer

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    query = request.form.get('query')
    if not query:
        return jsonify({'error': 'No query provided.'}), 400
    answer = generate_answer(query)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

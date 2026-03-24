import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from rag_engine import RAGEngine

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

ALLOWED_EXTENSIONS = {'csv', 'pdf', 'txt', 'xlsx', 'json'}
rag = RAGEngine()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status')
def status():
    return jsonify(rag.get_status())

@app.route('/api/models')
def models():
    try:
        import requests as req
        r = req.get('http://localhost:11434/api/tags', timeout=3)
        names = [m['name'] for m in r.json().get('models', [])]
        return jsonify({'models': names, 'count': len(names)})
    except Exception as e:
        return jsonify({'models': [], 'error': str(e)})

@app.route('/api/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': f'Unsupported file. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    try:
        result = rag.load_document(filepath, filename)
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get('question', '').strip()
    backend  = data.get('backend', 'ollama')
    if not question:
        return jsonify({'error': 'Empty question'}), 400
    if not rag.is_loaded():
        return jsonify({'error': 'No dataset loaded. Please upload a file first.'}), 400
    try:
        answer = rag.ask(question, backend=backend)
        return jsonify({'answer': answer, 'backend': backend})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/clear', methods=['POST'])
def clear():
    rag.clear()
    return jsonify({'status': 'cleared'})

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    print("\n🚀 DataChat running → http://localhost:5000\n")
    app.run(debug=True, port=5000)

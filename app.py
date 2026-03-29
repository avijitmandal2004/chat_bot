import os
import re
import pandas as pd
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'


# Global dataset
df = None


ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'json'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')



# Upload Dataset

@app.route('/api/upload', methods=['POST'])
def upload():

    global df

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'Empty filename'})

    if not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported file type'})

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)

        elif filename.endswith('.xlsx'):
            df = pd.read_excel(filepath)

        elif filename.endswith('.json'):
            df = pd.read_json(filepath)

        # Clean column names
        df.columns = df.columns.str.strip()

        # Clean string values
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].astype(str).str.strip().str.lower()

        return jsonify({
            "message": f"{filename} loaded successfully",
            "rows": len(df),
            "columns": list(df.columns)
        })

    except Exception as e:
        return jsonify({'error': str(e)})



# Chat API

@app.route('/api/chat', methods=['POST'])
def chat():

    global df

    data = request.json
    question = data.get('question', '').lower()

    if df is None:
        return jsonify({'answer': "No dataset loaded. Please upload a file first."})

    try:
        numeric_cols = df.select_dtypes(include='number').columns

        if "column" in question:
            return jsonify({'answer': f"Columns are: {', '.join(df.columns)}"})

        elif "row" in question:
            return jsonify({'answer': f"Total rows: {len(df)}"})

        elif "average" in question or "mean" in question:
            if len(numeric_cols) == 0:
                return jsonify({'answer': "No numeric columns found"})
            col = numeric_cols[0]
            return jsonify({'answer': f"Average of {col}: {df[col].mean():.2f}"})

        elif "max" in question:
            if len(numeric_cols) == 0:
                return jsonify({'answer': "No numeric columns found"})
            col = numeric_cols[0]
            return jsonify({'answer': f"Max of {col}: {df[col].max()}"})

        elif "min" in question:
            if len(numeric_cols) == 0:
                return jsonify({'answer': "No numeric columns found"})
            col = numeric_cols[0]
            return jsonify({'answer': f"Min of {col}: {df[col].min()}"})

        elif "sum" in question or "total" in question:
            if len(numeric_cols) == 0:
                return jsonify({'answer': "No numeric columns found"})
            col = numeric_cols[0]
            return jsonify({'answer': f"Total of {col}: {df[col].sum():.2f}"})

        elif "summary" in question:
            return jsonify({'answer': df.describe().to_string()})

        else:
            return jsonify({
                'answer': "Try asking: columns, rows, average, max, min, sum, summary"
            })

    except Exception as e:
        return jsonify({'answer': str(e)})



# Get Columns

@app.route('/api/columns', methods=['GET'])
def get_columns():

    global df

    if df is None:
        return jsonify([])

    return jsonify(list(df.columns))



# Get Unique Values

@app.route('/api/unique/<column>', methods=['GET'])
def get_unique(column):

    global df

    if df is None:
        return jsonify([])

    col_map = {c.lower(): c for c in df.columns}

    if column.lower() not in col_map:
        return jsonify([])

    real_column = col_map[column.lower()]

    values = df[real_column].dropna().unique().tolist()

    return jsonify(values)



# Filter API (FINAL FIX)

@app.route('/api/filter', methods=['POST'])
def filter_data():

    global df

    if df is None:
        return jsonify({'error': 'No dataset loaded'})

    data = request.json
    column = data.get('column', '').strip().lower()
    value = data.get('value', '').strip().lower()

    try:
        col_map = {c.lower(): c for c in df.columns}

        if column not in col_map:
            return jsonify({'error': f'Column "{column}" not found'})

        real_column = col_map[column]

        print("REAL VALUES:", df[real_column].unique())

        # Clean input
        value_clean = re.sub(r'[^a-z0-9]', '', value)

        # Clean dataset values
        clean_series = df[real_column].astype(str).str.replace(r'[^a-z0-9]', '', regex=True)

        # Match
        filtered_df = df[clean_series.str.contains(value_clean, na=False)]

        if filtered_df.empty:
            suggestions = df[real_column].unique()[:5].tolist()
            return jsonify({
                'error': f'No data found for {column} = {value}',
                'suggestions': suggestions
            })

        return jsonify({
            "rows": len(filtered_df),
            "data": filtered_df.head(20).to_dict(orient='records')
        })

    except Exception as e:
        return jsonify({'error': str(e)})


# Clear Dataset

@app.route('/api/clear', methods=['POST'])
def clear():

    global df
    df = None

    return jsonify({'status': 'cleared'})

# Run Server
if __name__ == '__main__':

    os.makedirs('uploads', exist_ok=True)

    print("\nDataChat running at http://localhost:5000\n")

    app.run(debug=True)

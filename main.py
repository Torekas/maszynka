from flask import Flask, render_template, request, session
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64


app = Flask(__name__)
app.secret_key = 'the random string'
df = None

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    global df
    if 'file' not in request.files:
        return 'No file uploaded.'

    file = request.files['file']

    if file.filename == '':
        return 'No selected file.'

    if file and allowed_file(file.filename):
        df = pd.read_csv(file) if file.filename.endswith('.csv') else pd.read_excel(file)
        return render_template('index.html', data=df.to_html(index=True, classes='table-scroll'))
    else:
        return 'Invalid file type. Only CSV and Excel files are allowed.'


def allowed_file(filename):
    allowed_extensions = ['csv', 'xlsx']
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.route('/describe', methods=['POST'])
def info():
    global df
    describe1 = df.describe()
    return render_template('index.html',describe= describe1.to_html(index=True, classes='table-scroll'))



@app.route('/information', methods=['POST'])
def information():
    global df
    summary = pd.DataFrame(df.dtypes, columns=['typ_danych'])
    summary['null_values'] = pd.DataFrame(df.isnull().any())
    summary['ile_null_values'] = pd.DataFrame(df.isnull().sum())
    summary['procent_nulls_values'] = round((df.apply(pd.isnull).mean() * 100), 2)
    summary.sort_index(inplace=True)
    return render_template('index.html',information= summary.to_html(index=True, classes='table-scroll'))


@app.route('/columns', methods=['POST'])
def columns():
    global df
    columns = df.columns.tolist()
    session['columns'] = columns
    return render_template('index.html', show_columns=True, columns=columns)

@app.route('/execute', methods=['POST'])
def execute():
    global df
    columns = df.columns.tolist()
    session['columns'] = columns

    selected_columns = request.form.getlist('column')

    if not selected_columns:
        return 'No columns selected.'

    plots = []

    for column in selected_columns:
        # Generate histogram
        plt.figure(figsize=(8, 6))
        plt.hist(df[column], bins='auto', color='blue')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.title(f'Histogram of {column}')
        plt.grid(True)

        # Convert the histogram plot to base64 for embedding in HTML
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        plots.append(plot_data)

        # Generate boxplot
        plt.figure(figsize=(8, 6))
        plt.boxplot(df[column].dropna())
        plt.xlabel(column)
        plt.ylabel('Value')
        plt.title(f'Boxplot of {column}')
        plt.grid(True)

        # Convert the boxplot to base64 for embedding in HTML
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        plots.append(plot_data)

    session['plots'] = plots

    return render_template('index.html', show_plots=True, selected_columns=selected_columns)

if __name__ == '__main__':
    app.run(debug=True)
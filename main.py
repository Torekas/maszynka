from flask import Flask, render_template, request, session
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import numpy as np
from sklearn.ensemble import IsolationForest
import seaborn as sns
from flask_session import Session


app = Flask(__name__)
app.secret_key = 'the random string'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)
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

@app.route('/describe_f', methods=['POST'])
def describe_f():
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
    selected_columns = request.form.getlist('column')
    session['selected_columns'] = selected_columns
    return render_template('index.html', show_columns=True, columns=columns)

selected_columns =[]

@app.route('/selected_columns_f', methods=['POST'])
def selected_columns_f():
    global selected_columns
    selected_columns = request.form.getlist('column')
    session['selected_columns'] = selected_columns
    return render_template('index.html')

@app.route('/remove_nulls_median', methods=['POST'])
def remove_nulls_median():
    global df
    global selected_columns

    for column in selected_columns:
        # Replace null values with the median
        median_value = df[column].median()
        df[column].fillna(median_value, inplace=True)

    summary = pd.DataFrame(df.dtypes, columns=['typ_danych'])
    summary['null_values'] = pd.DataFrame(df.isnull().any())
    summary['ile_null_values'] = pd.DataFrame(df.isnull().sum())
    summary['procent_nulls_values'] = round((df.apply(pd.isnull).mean() * 100), 2)
    summary.sort_index(inplace=True)
    return render_template('index.html', information_1=summary.to_html(index=True, classes='table-scroll'))


@app.route('/remove_nulls_mean', methods=['POST'])
def remove_nulls_mean():
    global df
    global selected_columns
    for column in selected_columns:
        # Replace null values with the mean
        mean_value = df[column].mean()
        df[column].fillna(mean_value, inplace=True)
    summary = pd.DataFrame(df.dtypes, columns=['typ_danych'])
    summary['null_values'] = pd.DataFrame(df.isnull().any())
    summary['ile_null_values'] = pd.DataFrame(df.isnull().sum())
    summary['procent_nulls_values'] = round((df.apply(pd.isnull).mean() * 100), 2)
    summary.sort_index(inplace=True)
    return render_template('index.html', information_1=summary.to_html(index=True, classes='table-scroll'))
@app.route('/plots', methods=['POST'])
def plots():
    global df
    global selected_columns
    columns = df.columns.tolist()
    session['columns'] = columns


    if not selected_columns:
        return 'No columns selected.'

    plots = []

    for column in selected_columns:
        # Generate histogram
        plt.figure(figsize=(4.5, 2))
        sns.histplot(df[column], bins='auto', kde=True, color='blue')
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
        plt.figure(figsize=(4.5, 2))
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

def calculate_iqr(column_data):
    Q1 = column_data.quantile(0.25)
    Q3 = column_data.quantile(0.75)
    return Q3 - Q1

# Identify outliers using the IQR method and isolation forest
@app.route('/outliers', methods=['POST'])
def outliers():
    global df
    global selected_columns
    contamination = request.form.get('contamination')
    random_state = request.form.get('random_state')

    outliers_info = []
    for col in selected_columns:

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        low_boundary = (Q1 - 1.5 * IQR)
        upp_boundary = (Q3 + 1.5 * IQR)
        num_of_outliers_L = (df[col] < low_boundary).sum()
        num_of_outliers_U = (df[col] > upp_boundary).sum()

        reshaped = np.array(df[col]).reshape(-1, 1)
        clf = IsolationForest(contamination=float(contamination), random_state=int(random_state))
        clf.fit(reshaped)
        predictions = clf.predict(reshaped)
        num_outliers = np.sum(predictions == -1)

        outliers_info.append({
            'column': col,
            'low': low_boundary,
            'upp': upp_boundary,
            'ile poniżej low': num_of_outliers_L,
            'ile powyżej upp': num_of_outliers_U,
            'isolation forest': num_outliers
        })
    outliers_df = pd.DataFrame(outliers_info)

    return render_template('index.html', outliers = outliers_df.to_html( classes='table-scroll'))

#ostatnio robiłeś outliersy , wykrywanie iqr działa, musisz teraz dodać metody usuwania i tak samo co jesli bedą jakieś wartości nullowe plus KDE line dodaj do boxplotów



if __name__ == '__main__':
    app.run(debug=True)
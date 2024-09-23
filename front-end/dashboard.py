import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px
import requests

# Load model history
history = pd.read_csv('../history.csv')

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Line plot for accuracy
fig_accuracy = px.line(history, x='epochs', y=['accuracy', 'val_accuracy'],
                       labels={'value': 'Accuracy', 'variable': 'Dataset'}, title="Model Accuracy")

# Line plot for loss
fig_loss = px.line(history, x='epochs', y=['loss', 'val_loss'],
                   labels={'value': 'Loss', 'variable': 'Dataset'}, title="Model Loss")

# Define Dash layout
app.layout = dbc.Container([
    html.H1("Student Performance Prediction Dashboard"),

    dbc.Row([
        dbc.Col(dcc.Graph(figure=fig_accuracy), width=6),
        dbc.Col(dcc.Graph(figure=fig_loss), width=6),
    ]),

    html.Div([
        html.H2("Input Features"),
        dcc.Input(id='student-age', type='number', placeholder='Student Age'),
        dcc.Input(id='additional-work', type='number', placeholder='Additional Work'),
        dcc.Input(id='total-salary', type='number', placeholder='Total Salary'),
        dcc.Input(id='weekly-study-hours', type='number', placeholder='Weekly Study Hours'),
        dcc.Input(id='attendance-classes', type='number', placeholder='Attendance to Classes'),
        dcc.Input(id='preparation-exams', type='number', placeholder='Preparation to Midterms'),
        dcc.Input(id='cumulative-gpa', type='number', placeholder='Cumulative GPA'),
        dcc.Input(id='scholarship-type', type='text', placeholder='Scholarship Type'),
        dcc.Input(id='regular-activity', type='text', placeholder='Regular Activity'),

        dbc.Button('Submit', id='submit-button', n_clicks=0, color='primary', className='mt-3'),
    ]),

    html.Div(id='prediction-output', className='mt-4')
])


@app.callback(
    Output('prediction-output', 'children'),
    Input('submit-button', 'n_clicks'),
    Input('student-age', 'value'),
    Input('additional-work', 'value'),
    Input('total-salary', 'value'),
    Input('weekly-study-hours', 'value'),
    Input('attendance-classes', 'value'),
    Input('preparation-exams', 'value'),
    Input('cumulative-gpa', 'value'),
    Input('scholarship-type', 'value'),
    Input('regular-activity', 'value')
)
def update_prediction(n_clicks, student_age, additional_work, total_salary,
                      weekly_study_hours, attendance_classes, preparation_exams,
                      cumulative_gpa, scholarship_type, regular_activity):
    if n_clicks > 0:
        input_data = {
            'Student Age': student_age,
            'Additional work': additional_work,
            'Total salary if available': total_salary,
            'Weekly study hours': weekly_study_hours,
            'Attendance to classes': attendance_classes,
            'Preparation to midterm exams 1': preparation_exams,
            'Cumulative grade point average in the last semester (/4.00)': cumulative_gpa,
            'Scholarship type': scholarship_type,
            'Regular artistic or sports activity': regular_activity
        }

        # Make the POST request to the API
        response = requests.post('http://127.0.0.1:5000/predict', json=input_data)

        if response.status_code == 200:
            prediction = response.json()
            return f'Prediction: {prediction["prediction"]}'
        else:
            return f'Error: {response.text}'
    return ''


if __name__ == '__main__':
    app.run_server(debug=True)

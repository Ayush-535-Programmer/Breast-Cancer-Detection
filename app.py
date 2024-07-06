from flask import Flask, render_template, request, redirect
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model\\breast_cancer_detector.pickle','rb'))

def run_model(inputFeatures):
    features_names = ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
        'mean smoothness', 'mean compactness', 'mean concavity',
        'mean concave points', 'mean symmetry', 'mean fractal dimension',
        'radius error', 'texture error', 'perimeter error', 'area error',
        'smoothness error', 'compactness error', 'concavity error',
        'concave points error', 'symmetry error', 'fractal dimension error',
        'worst radius', 'worst texture', 'worst perimeter', 'worst area',
        'worst smoothness', 'worst compactness', 'worst concavity',
        'worst concave points', 'worst symmetry', 'worst fractal dimension']

    input_features= inputFeatures

    features_value = [np.array(input_features)]

    df = pd.DataFrame(features_value,columns=features_names)
    output = model.predict(df)
    
    if output == 0:
        return "This patient has **Breast cancer**"
        
    return "Great No Breast Cancer Symptom recognised"



@app.route('/', methods = ['GET', 'POST'])
def main_page():
    report = "Null"
    if request.method == 'POST':
        try:
            inputs = [float(x) for x in request.form.values()]
            print(inputs)
            report = run_model(inputs)
        except :
            report = "Please Fill all values Properly"
    return render_template('index.html', report = report)



if __name__ == "__main__":
    app.run()

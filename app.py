from flask import Flask, render_template, request
import pickle
import gzip
import numpy as np
from PIL import Image
with gzip.open('modelh.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)




@app.route('/')
def index():
    return render_template('health.html')

@app.route('/predict', methods = ['POST'])
def recomend_result():
    age = float(request.form.get('age',0))
    anaemia = float(request.form.get('anaemia',0))
    creatinine_phosphokinase= float(request.form.get('creatinine_phosphokinase',0))
    diabetes = float(request.form.get('diabetes',0))
    ejection_fraction = float(request.form.get('ejection_fraction',0))
    high_blood_pressure = float(request.form.get('high_blood_pressure',0))
    platelets = float(request.form.get('platelets',0))
    serum_creatinine=float(request.form.get('serum_creatinine',0))
    serum_sodium=float(request.form.get('serum_sodium',0))
    sex=float(request.form.get('sex',0))
    smoking=float(request.form.get('smoking',0))
    time=float(request.form.get('time',0))

    # prediction
    death = ['0','1']
    result = model.predict(np.array([age,	anaemia,	creatinine_phosphokinase,	diabetes,	ejection_fraction,	high_blood_pressure,	platelets,	serum_creatinine,	serum_sodium,	sex,	smoking,	time]).reshape(1,12))
    index = result[0] - 1
    result = str(death[index])
    return render_template('health.html', result=result)


if __name__ == '__main__':
    app.run(debug = True,port=5001)

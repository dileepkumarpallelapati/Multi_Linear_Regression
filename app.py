import pickle
from flask import Flask,render_template,request
import numpy as np
import pandas as pd


with open('Multi.pkl','rb') as f:
    model=pickle.load(f)


app=Flask(__name__)

@app.route('/')
def home():
    return (render_template('index.html'))


@app.route('/predict',methods=['POST','GET'])



def prediction():
    rnd=float(request.form.get('rnd'))
    admin=float(request.form.get('admin'))
    marketing=float(request.form.get('marketing'))
    state=float(request.form.get('state'))


    #features=np.array([[rnd,admin,marketing,state]])
    features = pd.DataFrame(
        [[rnd, admin, marketing, state]],
        columns=['R&D Spend', 'Administration', 'Marketing Spend', 'State']
    )

    prediction_text=model.predict(features)

    result=f'{prediction_text[0]}'

    return render_template('index.html',prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)
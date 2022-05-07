import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from flask_heroku import Heroku
app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))
heroku = Heroku(app)
@app.route('/')
def home():
    return render_template('index.html')
iris_target = ['setosa', 'versicolor', 'virginica']
@app.route('/predict',methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    x_test = [np.array(features)]
    prediction = model.predict(x_test)

    return render_template('index.html',prediction_text = iris_target[prediction[0]])

if __name__=='__main__':
    app.run(debug=True)

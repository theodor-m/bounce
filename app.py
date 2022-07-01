import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPRegressor

from sklearn import metrics

from sklearn.model_selection import GridSearchCV

from flask import Flask, render_template, request

app = Flask(__name__)
model = pickle.load(open('model_old.pkl', 'rb'))
model2 = pickle.load(open('model_twoinn.pkl', 'rb'))


@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/')
def base():
    return render_template('home.html')


@app.route('/bounce')
def bounce():
    return render_template('bounce.html')


@app.route('/bounce2')
def bounce2():
    return render_template('bounce_2.html')


@app.route('/drop')
def drop():
    return render_template('drop.html')


@app.route('/predict2', methods=['POST'])
def predict2():
    """Grabs the input values and uses them to make prediction"""
    T1 = float(request.form["T1"])
    T2 = float(request.form["T2"])

    df = pd.read_csv('bounce2.csv')
    x = df.drop('H', axis=1)
    y = df['H']
    trainX, testX, trainY, testY = train_test_split(x.to_numpy(), y.to_numpy(), test_size=0.2)

    sc = StandardScaler()
    scaler = sc.fit(trainX)

    array = [T1, T2]
    array = np.array(array)
    array_scaled = scaler.transform(array.reshape(1, -1))

    prediction = model2.predict(array_scaled)  # this returns a list e.g. [127.20488798], so pick first element [0]
    output = round(prediction[0], 2)

    return render_template('bounce_2.html',
                           prediction_text=f' Prediction: {output}')


@app.route('/predict', methods=['POST'])
def predict():
    """Grabs the input values and uses them to make prediction"""
    T1 = float(request.form["T1"])
    T2 = float(request.form["T2"])
    T3 = float(request.form["T3"])

    df = pd.read_csv('bounce.csv')
    x = df.drop('H', axis=1)
    y = df['H']
    trainX, testX, trainY, testY = train_test_split(x.to_numpy(), y.to_numpy(), test_size=0.2)

    sc = StandardScaler()
    scaler = sc.fit(trainX)

    array = [T1, T2, T3]
    array = np.array(array)
    array_scaled = scaler.transform(array.reshape(1, -1))

    prediction = model.predict(array_scaled)  # this returns a list e.g. [127.20488798], so pick first element [0]
    output = round(prediction[0], 2)

    return render_template('bounce.html',
                           prediction_text=f' Prediction: {output}')


@app.route('/predictdrop', methods=['POST'])
def predictdrop():
    """Grabs the input values and uses them to make prediction"""
    T1 = float(request.form["T1"])
    T2 = float(request.form["T2"])

    T1 = T1 * 2 - 0.06

    df = pd.read_csv('bounce2.csv')
    x = df.drop('H', axis=1)
    y = df['H']
    trainX, testX, trainY, testY = train_test_split(x.to_numpy(), y.to_numpy(), test_size=0.2)

    sc = StandardScaler()
    scaler = sc.fit(trainX)

    array = [T1, T2]
    array = np.array(array)
    array_scaled = scaler.transform(array.reshape(1, -1))

    prediction = model2.predict(array_scaled)  # this returns a list e.g. [127.20488798], so pick first element [0]
    output = round(prediction[0], 2)

    return render_template('drop.html',
                           prediction_text=f' Prediction: {output}')


if __name__ == '__main__':
    app.run()

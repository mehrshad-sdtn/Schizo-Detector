from flask import Flask, render_template, request
from tensorflow import keras
import tensorflow as tf
import eeg_processing as eegp
import image_utils
import os
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

model = model = keras.models.load_model('simple_model_final.h5')

def clear_directory(dir_path):
    files = os.listdir(dir_path)
    for f in files:
        os.remove(os.path.join(dir_path, f))


def diagnose(zeros, ones):
    percent_sch = round((ones/12)*100, 1)
    print(percent_sch)
    if percent_sch > 60 and percent_sch < 90:
        return f"You are at a very high ({percent_sch}%) risk of having Schizophrenia, please consult a professional for furthur diagnosis and assistance as soon as possible"

    elif percent_sch > 90:
        return f"You are diagnosed with Schizophrenia (with {percent_sch}% confidence), please seek a professional's help for your treatment options as soon as possible"

    elif percent_sch < 60 and percent_sch > 40:
        return f"You are at a the ({percent_sch}%) risk of having Schizophrenia, please consult a professional for furthur diagnosis and assistance"

    elif percent_sch < 40 and percent_sch > 10:
        return f"You probably do not have Schizophrenia, But still have a {percent_sch}% risk, please also consult a professional"

    else:
        return f"You are healthy ({percent_sch}% chance of having Schizophrenia)"



@app.route('/', methods=['GET'])
def hello_world():
    clear_directory('EEG')
    clear_directory('images')
    return render_template('index.html')


@app.route('/diagnosis', methods=['POST'])
def predict():
    eegfile=request.files['eegfile']
    eegpath = os.path.join("EEG", eegfile.filename)
    eegfile.save(eegpath)
    eegp.process_eeg_data()
    X = image_utils.image_arrays_from_directory('images')
    y_preds = model.predict(X)
    predictions = np.argmax(y_preds, axis=-1)
    unique, counts = np.unique(predictions, return_counts=True)
    dictionary = dict(zip(unique, counts))

    zeros, ones = 0, 0
    if 0 in dictionary:
        zeros = dictionary[0]    
    else:
        zeros = 0
    
    if 1 in dictionary:
        ones = dictionary[1]
    else:
        ones = 0

    print(zeros, ones)
    diagnosis = diagnose(zeros, ones)
    print(diagnosis)
    return render_template('sub.html', text=diagnosis)
    

if __name__ == '__main__':
    app.run(port=3000, debug=True) 
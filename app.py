from flask import Flask, render_template, request
import os
import numpy as np
import classifier as clf
import sys
import logging
import PIL

app = Flask(__name__)
app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)

app.config['UPLOAD_FOLDER'] =os.path.join('static', 'images')

def build_and_save_image(image):
    data = image.read()
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
    with open(file_path, 'wb') as file:
        file.write(data)

def clean_image_dir():
    for im in os.listdir(app.config['UPLOAD_FOLDER']):
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], im))


@app.route("/")
def hello():
    return render_template("index.html")

@app.route("/sub", methods = ['POST'])
def submit():
    predictions = []
    clean_image_dir()
    if request.method == "POST":
        image = request.files["file"]
        build_and_save_image(image)
        im_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
        pred = clf.get_image_prediction(im_path)

        return render_template('sub.html', name = pred, image=im_path)

if __name__ == '__main__':
    app.run()
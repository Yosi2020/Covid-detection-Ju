from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import cv2

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

from PIL import Image
import PIL

app = Flask(__name__)

model_path = "./model/my_model.h5"

model = load_model(model_path)

#model._make_predict_function()  

classes={
    0:'Covid-19', 
    1:'Normal', 
    2:'Pneumonia'
    }

print('Model loaded. Check http://127.0.0.1:5000/')

def model_predict(img_path, model):
	Image_size = 224
	img = image.load_img(img_path)
	img = np.array(img)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = cv2.resize(img, (Image_size,Image_size))
	img = img.astype("float") / 255.0

	x = np.expand_dims(img, axis=0)

	preds = model.predict(x)

	return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
	if request.method == 'POST':

		# Get the file from post request
		f = request.files['file']

		# Save the file to ./uploads
		basepath = os.path.dirname(__file__)
		file_path = os.path.join(
			basepath, 'uploads', secure_filename(f.filename))
		f.save(file_path)

		# Make a prediction
		prediction = model_predict(file_path, model)

		# Process your result for human
		# pred_class = preds.argmax(axis=-1)            # Simple argmax
		Value1 = f"{prediction[0][0]}"
		Value2 = f"{prediction[0][1]}"
		Value3 = f"{prediction[0][2]}"
		result = f"{classes[int(np.argmax(prediction[0]))]}"
		print(Value1)
		print(Value2)
		print(Value3)
		print('We think that is {}.'.format(result))
		Pred_text = "***************************************************************************************************** Covid-19 {:}  ***************************************************************************************************** Normal {:} ****************************************************************************************************** Pneumonia {:} ********************************************************************************************************* We think that your x-ray final result is :- {:}".format(Value1,Value2,Value3,result)
		return Pred_text
	return None


if __name__ == '__main__':
    app.run(debug=True)
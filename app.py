import os
import cv2
from PIL import Image as im
import numpy as np
from utils import dict_to_ndarray
from flask import Flask, render_template, request, send_from_directory

from tensorflow.keras import models

app = Flask(__name__)

class DigitClassifier:
    def __init__(self, image = None) -> None:
        self.image = image or np.zeros((20, 20, 20)) 
        self.model = models.load_model('digit_recognizer_2.h5')
        print('Model Loaded Sucessfully!')
    
    def show(self):
        cv2.imshow('image', cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        
    def update(self, image, width, height):
        self.image = image
        self.width = width
        self.height = height
    
    @staticmethod
    def reduced_image(image, width, height):
        assert width >= 50 and height >= 50 ,"Cannot reduce more than this"
        image  = cv2.GaussianBlur(image,(9,15),cv2.BORDER_DEFAULT)
        scale_percent = 10 # percent of original size
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        reduced = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        reduced =cv2.cvtColor(reduced, cv2.COLOR_BGR2GRAY)
        return reduced, reduced.shape

    def show_reduced(self):
        reduced, _ = DigitClassifier.reduced_image(np.array(self.image), self.width, self.height)
        cv2.imshow('reduced', reduced)
        cv2.waitKey(0)

    def save_img(self):
        self.image('test.png')

    def save_reduced(self):
        reduced, _ = DigitClassifier.reduced_image(np.array(self.image), self.width, self.height)
        im.fromarray(reduced).save('reduced.png')


    def predict(self):
        image, reduced_shape  = DigitClassifier.reduced_image(np.array(self.image), self.width, self.height)

        print(image.shape)
        prediction = self.model.predict(image.reshape(1, reduced_shape[0] , reduced_shape[1]))

        print(np.argmax(prediction))

digit_classifier = DigitClassifier()



# Receiving Data
@app.post('/postimage')
def post_data():
    global digit_classifier
    request_data = request.get_json()
    width, height = request_data['width'],request_data['height']
    image = dict_to_ndarray(request_data['data'],width, height , 4)
    digit_classifier.update(im.fromarray(image.astype(np.uint8)), width, height)
    digit_classifier.save_reduced()
    digit_classifier.predict()

    return "200"

# Initiating Web App
@app.route("/")
def index():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),'favicon.ico')




if __name__ == '__main__':
    app.run()


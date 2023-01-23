from PIL import Image as im
import numpy as np
from flask import Flask, render_template, request, send_from_directory

from tensorflow.keras import models
import cv2

app = Flask(__name__)

model = models.load_model('output/digit_recognizer_2.h5')

def reduce_img(img):
    img_3 = img[:, :, :-1]
    print(img.shape)

    gray_img = cv2.cvtColor(img_3, cv2.COLOR_RGB2GRAY)
    gray_img_resized = cv2.resize(gray_img, (28,28))
    print(gray_img.shape)

def predict(img: im.Image) -> int:

    
    img = img.resize((28,28))


    img = img.convert('L')

    img = np.asarray(img)


    img = img/255.0

    res = model.predict(img.reshape(1, 28, 28, 1))


    print(np.argmax(res))

# Receiving Data
@app.post('/postimage')
def post_data():
    request_data = request.get_json()
    width, height = request_data['width'],request_data['height']
    # image = im.fromarray(np.array([int(x) for _ , x in request_data['data'].items()]).resize(width, height, 4))

    img = np.array([x for _,x in request_data['data'].items()]).reshape((width, height, 4))
    print(img.shape)

    reduce_img(img)

    return "200"

# Initiating Web App
@app.route("/")
def index():
    return render_template('index.html')





if __name__ == '__main__':
    app.run()


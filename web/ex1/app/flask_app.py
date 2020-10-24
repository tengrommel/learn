from flask import Flask, request
import numpy as np
from PIL import Image
import imageio
from tensorflow.keras.models import model_from_json
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import re
import base64
# The base64 module is needed to decode the string and then the file is saved as image.png

json_file = open('model.json', 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("weights.h5")
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
graph = tf.compat.v1.get_default_graph()

app = Flask(__name__)


# We might sometimes get images in the form of base64 encoded strings
# if the user makes an image POST request with a suitable setting.
def convert_image(img_data):
    # img_str = re.search(r'base64,(.*)', str(img_data)).group(1)
    with open('output.png', 'wb') as output:
        output.write(base64.b64decode(img_data))


@app.route("/")
def index():
    return "Oops, nothing here!"


@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    global model, graph
    img_data = request.get_data()
    convert_image(img_data)
    x = imageio.imread('output.png')
    x = Image.fromarray(x).resize((28, 28))
    x = np.reshape(x, (1, 28, 28, 1))
    with graph.as_default():
        # perform the prediction
        out = model.predict(x)
        print(out)
        print(np.argmax(out, axis=1))
        # convert the response to a string
        response = np.argmax(out, axis=1)
        return str(response[0])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)

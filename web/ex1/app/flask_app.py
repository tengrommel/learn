from flask import Flask, request
import numpy as np
from PIL import Image
import imageio
from tensorflow.keras.models import model_from_json
import tensorflow as tf
from tensorflow.keras import backend

json_file = open('model.json', 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("weights.h5")
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
graph = tf.compat.v1.get_default_graph()

app = Flask(__name__)


@app.route("/")
def index():
    return "Hello World!"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)

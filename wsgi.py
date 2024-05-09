############################
# @author Elias De Hondt   #
# @see https://eliasdh.com #
# @since 09/05/2024        #
############################
# FUNCTIE: This file contains the code for the Flask web server

import tflite_runtime.interpreter as tflite                 # type: ignore
from flask import Flask, Response, render_template, request # type: ignore
import numpy as np                                          # type: ignore
import json                                                 # type: ignore

class MNIST:
    """
    The MNIST model
    """
    def __init__(self):
        """
        Initialize the MNIST model
        """
        self.interpreter = tflite.Interpreter(model_path = "mnist_model.tflite")
        self.interpreter.allocate_tensors()

    def predict_digit(self, image):
        """
        Predict the digit in the image
        :param image: The image data
        :return: The predicted digit
        """
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        self.interpreter.set_tensor(input_details[0]["index"], np.array([np.expand_dims(image, axis=-1)], dtype=np.float32))
        self.interpreter.invoke()
        return self.interpreter.get_tensor(output_details[0]["index"])

app = Flask(__name__, template_folder = "static")

def normalize(image_data):
    """
    Normalize the image data to values between 0 and 1
    :param image_data: The image data to normalize
    :return: The normalized image data
    """
    return [[x / 255.0 for x in y] for y in image_data]

def rearrange_image_data(image):
    """
    Rearrange the image data to a 28x28 array
    :param image: The image data to rearrange
    :return: The rearranged image data
    """
    image_array = np.array(list(image.values()))
    grey_image_array = image_array[3::4].reshape(28, 28)
    return grey_image_array

@app.route('/')
def interface():
    """
    Render the interface
    :return: The rendered interface
    """
    return render_template("interface.html")

@app.route('/predict_digit', methods = ["POST"])
def predict_digit():
    """
    Predict the digit in the image
    :return: The predicted digit
    """
    results = MNIST().predict_digit(normalize(rearrange_image_data(json.loads(request.form["image_data"])))).tolist()
    json_response = json.dumps({"results": json.dumps(results)})
    return Response(json_response, mimetype = "text/json")

if __name__ == "__main__": app.run(debug = True)
# USAGE
# Start the server:
# 	python run_keras_server.py
# Submit a request via cURL:
# 	curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
#	python simple_request.py

# https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html

# import the necessary packages
from keras.models import load_model 
from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing import image
from PIL import Image
import numpy as np
import flask
import io

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None

def load_model():
	# load the pre-trained Keras model (here we are using a model
	# pre-trained on ImageNet and provided by Keras, but you can
	# substitute in your own networks just as easily)
	global model #load models here
	#model =load_model('emotion_model.hdf5')
	model = ResNet50(weights="imagenet")
	return model

def prepare_image(image, target):
	# if the image mode is not RGB, convert it
	if image.mode != "RGB":
		image = image.convert("RGB")

	# resize the input image and preprocess it
	image = image.resize(target)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	image = imagenet_utils.preprocess_input(image)

	# return the processed image
	return image

@app.route("/")
def hello():
	return "Running Image Classification Server!"
 
# This error is due to the fact that your browser is performing a GET request, 
# but /predict only accepts a POST. cCURL Requests
@app.route("/predict", methods=["GET"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "GET":
        #if flask.request.files.get("image"):
            # read the image in PIL format
        model = ResNet50(weights='imagenet')
        img_path = 'dog.jpg'
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        preds = model.predict(x)
        results = imagenet_utils.decode_predictions(preds)
        data["predictions"] = []

            # loop over the results and add them to the list of
            # returned predictions
        for (imagenetID, label, prob) in results[0]:
            r = {"label": label, "probability": float(prob)}
            data["predictions"].append(r)

            # indicate that the request was a success
        data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))
	#load_model()
	print ('app is running')
	app.run()


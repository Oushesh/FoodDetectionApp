
from keras.models import load_model 
from keras.applications import ResNet50
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from PIL import Image
import numpy as np
import flask
import io


model = ResNet50(weights='imagenet')

img_path = 'dog.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
print('Predicted:', decode_predictions(preds))


#Decode the dictionary here:


#for (imagenetID, label, prob) in decode_predictions(preds)[0]:
#	print ("imagenet:" ,imagenetID, "label:", label, "prob:", prob)

output=[]
for (imagenetID, label, prob) in  decode_predictions(preds)[0]:
    #print ("imagenet:" ,imagenetID, "label:", label, "prob:", prob)
    r = {"label": label, "probability": float(prob)}
    output.append(r)

#print (output)
print (flask.jsonify(output))
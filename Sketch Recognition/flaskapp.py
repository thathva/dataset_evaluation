

from flask import Flask
import base64
import numpy as np
import io
import os
from PIL import Image
from flask import request
from flask import jsonify
from keras.models import Sequential
import keras
import tensorflow as tf
from keras.models import load_model
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense
from keras.layers import MaxPooling2D,Conv2D,Flatten
from keras.preprocessing.image import img_to_array


app=Flask(__name__)

def get_model():
    global classifier,graph
    classifier=load_model('my_model.h5')
    print("loaded")
    graph = tf.get_default_graph()
    
def preprocess_image(image,target_size):
    if image.mode!="RGB":
        image=image.convert("RGB")
    image=image.resize(target_size)
    image=img_to_array(image)
    image=np.expand_dims(image,axis=0)

    return image

get_model()
@app.route('/',methods=["POST"])
def predict():
    message=request.get_json(force=True)
    encoded=message['image']
    decoded=base64.b64decode(encoded)
    with graph.as_default(): 
        image=Image.open(io.BytesIO(decoded))
        processed_image=preprocess_image(image,target_size=(64,64))
        prediction=classifier.predict_classes(processed_image).tolist()
    
        response={
            'prediction': {
                'person':prediction
                
            }
    }
    
    return jsonify(response)
if __name__=='__main__':
    app.debug=True
    app.run(host='0.0.0.0',port=5000)

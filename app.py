#import libraries & dependencies
import os
import io
import numpy as np
import keras
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from keras.applications.xception import (Xception, preprocess_input, decode_predictions)
from keras import backend as K
from flask import Flask, request, redirect, url_for, jsonify

#define flaskapp
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Uploads'
model = None
graph = None

#functions
def load_model():
    global model
    global graph
    model = Xception(weights="imagenet")
    graph = K.get_session().graph
def prepare_image(img):
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    #return processed image
    return img

#call functions
load_model()

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    data = {"success": False}
    if request.method == 'POST':
        if request.files.get('file'):
            #read input file
            file = request.files['file']
            #read filename
            filename = file.filename
            #create os path to uploads directory
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            #load the saved image and resize to the Xception 299x299 pixels
            image_size = (299, 299)
            im = keras.preprocessing.image.load_img(filepath,target_size=image_size,grayscale=False)
            #preprocess the image for classification
            image = prepare_image(im)
            global graph
            with graph.as_default():
                preds = model.predict(image)
                res = decode_predictions(preds)
                #print the res
                print(res)
                data["predictions"] = []
                #loop over the results and add to returned predictions
                for (imagenetID, label, prob) in res[0]:
                    r = {"label": label, "probability": float(prob)}
                    data["predictions"].append(r)
                #store boolean for process success
                data["success"] = True
        return jsonify(data)

    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''


if __name__ == "__main__":
    app.run(debug=True)

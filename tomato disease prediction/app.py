from __future__ import division, print_function
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

# define flask app
app = Flask(__name__)

# model name
MODEL_PATH = 'models/inceptionV3_model.h5'

# load trained model
model = load_model(MODEL_PATH)

def model_predict(img_path, model):
    print('Uploaded image path: ',img_path)
    loaded_image = image.load_img(img_path, target_size=(224, 224))

    # preprocess the image
    loaded_image_in_array = image.img_to_array(loaded_image)

    # normalize
    loaded_image_in_array=loaded_image_in_array/255
    from tensorflow.keras.applications.inception_v3 import preprocess_input
    # add additional dim such as to match input dim of the model architecture
    x = np.expand_dims(loaded_image_in_array, axis=0)
    x = preprocess_input(x)

    # prediction
    prediction = model.predict(x)

    results=np.argmax(prediction, axis=1)

    if results==0:
        results="The leaf is Tomato Bacterial Spot"
    elif results==1:
        results="The leaf is Tomato Early Blight"
    elif results==2:
        results="The leaf is Tomato Healthy"
    elif results==3:
        results="The leaf is Tomato Late Blight"
    elif results==4:
        results="The leaf is Tomato Leaf Mold"
    elif results == 5:
        results = "The leaf is Tomato Septoria leaf spot"
    elif results == 6:
        results = "The leaf is Tomato Spider Mites Two-spotted Spider Mite"
    elif results == 7:
        results = "The leaf is Tomato Target Spot"
    elif results == 8:
        results = "The leaf is Tomato Mosaic Virus"
    else:
        results="The leaf is Tomato Yellow Leaf Curl Virus"

    return results

@app.route('/', methods=['GET'])
def index():
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

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None

from pyngrok import ngrok

if __name__ == '__main__':
    app.run(port=5001,debug=True)
    url = ngrok.connect(port=8501)
    print(url)

from flask import Flask, render_template, request
import numpy as np
from PIL import Image
import os
from werkzeug.utils import secure_filename
import io
from texture import lbp_features,Euclidean_distance,distances1
#from histogram import thisdictcolor,distances
import imageio.v2 as imageio

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = "static/"

@app.route('/')
def upload_file():
    #print(thisdictcolor)
    return render_template('index.html')


@app.route('/display', methods = ['GET', 'POST'])
def save_file():
    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)
        f.save(app.config['UPLOAD_FOLDER'] + filename)
        image=imageio.imread("C:/Users/asus/Desktop/mini_projet_data_science/static/"+ filename)
        texture = lbp_features(image,2,8)
        result=distances1(texture)
        print(result)
        os.remove("C:/Users/asus/Desktop/mini_projet_data_science/static/"+ filename)
    return render_template('index.html', result=result) 
    

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug = True,use_reloader=False)
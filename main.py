import os
import numpy as np
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from fastai.vision import *
from PIL import Image
import base64
import io


app = Flask(__name__)
	
@app.route('/')
def upload_form():
	return render_template('upload.html')

@app.route('/sub', methods=['POST','GET'])
def upload_image():
        file = request.files['file']
        filename = secure_filename(file.filename)
        data = request.files['file'].read()
        byte_stream = io.BytesIO(data)
        learn = load_learner("")
        image_path = open_image(byte_stream)
        pred_fastai = learn.predict(image_path)
        p=np.array(pred_fastai[1],np.uint8)
        p=p.reshape(256,256)
        # img = cv2.imdecode(p,cv2.IMREAD_COLOR)
        img = Image.fromarray((p * 255).astype(np.uint8))
        data = io.BytesIO()
        img.save(data, 'jpeg')
        encoded_img_data = base64.b64encode(data.getvalue())
        return render_template('upload.html', img_data=encoded_img_data.decode('utf-8'))

if __name__ == "__main__":
    app.run()
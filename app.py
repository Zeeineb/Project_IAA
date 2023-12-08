from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import decode_predictions
import numpy as np

app = Flask(__name__)

base_dir = 'C:/Users/ASUS/Desktop/Projet IAA - Last version'
static_folder = 'static'

app.config['UPLOAD_FOLDER'] = os.path.join(base_dir, static_folder, 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

img_size = (256, 256)

model = tf.keras.models.load_model('C:/Users/ASUS/Desktop/Projet IAA - Last version/MyModel')

def process_image(file_path):
    img = image.load_img(file_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predictions = model.predict(img_array)

    class_labels = ['COVID', 'NORMAL', 'PNEUMONIA']

    decoded_predictions = [class_labels[i] for i in np.argmax(predictions, axis=1)]

    return decoded_predictions[0]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            prediction, accuracy = process_image(file_path)

            return render_template('results.html', prediction=prediction, filename=filename)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
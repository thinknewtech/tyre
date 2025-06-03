from flask import Flask, request, jsonify
from flask_restful import Resource, Api
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# Load model once when the app starts
model = tf.keras.models.load_model('best_model.h5')

# Preprocessing function
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = tf.keras.applications.resnet.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Flask setup
app = Flask(__name__)
api = Api(app)

# RESTful Resource
class Predict(Resource):
    def post(self):
        if 'image' not in request.files:
            return {"error": "No image file provided"}, 400
        
        img_file = request.files['image']
        if img_file.filename == '':
            return {"error": "Empty filename"}, 400

        # Save uploaded image to a temp path
        filepath = os.path.join("temp", img_file.filename)
        os.makedirs("temp", exist_ok=True)
        img_file.save(filepath)

        try:
            img_array = preprocess_image(filepath)
            prediction = model.predict(img_array)[0][0]
            label = "good" if prediction > 0.5 else "defective"

            return {
                "prediction": label,
                "confidence": f"{round(prediction * 100, 2)}%"
            }

        finally:
            # Clean up temporary image
            os.remove(filepath)

# Route setup
api.add_resource(Predict, '/predict')

if __name__ == '__main__':
    app.run(debug=True)

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import numpy as np

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename


    def predict(self):
        # load model
        model = load_model(os.path.join("model", "model.h5"))

        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (224,224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = np.argmax(model.predict(test_image), axis=1)
        print(result)

        if result[0] == 1:
            prediction = 'Fresh Leaf'
            return [{ "image" : prediction}]
        else:
            prediction = 'Disease Leaf'
            return [{ "image" : prediction}]
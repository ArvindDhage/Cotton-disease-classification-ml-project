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
            des_type = "Not to apply any pesticides"
            return [{ "image" : prediction,
                     "solution" : des_type}]
        else:
            des_type = {"Fusarium" : "including crop rotation, resistant cultivars, and appropriate fungicides, can help mitigate Fusarium-related diseases in cotton plants.",
                        "Cercospora": "Adopting cultural practices such as proper irrigation, crop rotation, and timely fungicide applications is essential for managing Cercospora leaf spot in cotton plants.",
                        "Alternaria": "Control Alternaria leaf spot in cotton by implementing proper sanitation measures, using disease-resistant cultivars, and applying fungicides as a part of an integrated disease management strategy."}
            disease_type = ["Fusarium", "Cercospora", "Alternaria"]
            prediction = 'Disease Leaf'
            return [{"image" : prediction,
                     "solution for disease" : des_type,
                     "disease_type" : disease_type}]
        
    

        

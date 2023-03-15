import onnxruntime
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from utils.preprocessing import preprocess_numpy

class OnnxModel:
    def __init__(self, model_path):
        self.session = onnxruntime.InferenceSession(model_path)
    
    def predict(self, input_data):
        input_data = preprocess_numpy(input_data).numpy().reshape(1, 3, 224, 224)
        output = self.session.run([], {self.session.get_inputs()[0].name: input_data})

        return output

model = OnnxModel("models/mtailor.onnx")

img1 = Image.open('./n01667114_mud_turtle.jpeg')


output = model.predict(img1)
print(np.argmax(output))

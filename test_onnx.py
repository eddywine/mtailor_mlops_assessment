#import torch
import onnx
import onnxruntime
import numpy as np
from PIL import Image
import pytorch_model

ONNX_FILENAME = "mtailor.onnx"

# Load the ONNX model
session = onnxruntime.InferenceSession(ONNX_FILENAME)

# Calling the defined Pytorch model to access the preprocessing step of the image
mtailor_model = pytorch_model.Classifier(pytorch_model.BasicBlock, [2, 2, 2, 2])


def testing_validity_onnx_model():
    """Summary: This function tests the validity of the loaded onnx model.
    """
    # Load the exported model
    onnx_model = onnx.load(ONNX_FILENAME)

    # The following is a function to check whether the loaded onnx model is valid or not
    try:
        onnx.checker.check_model(onnx_model)
        print("{ONNX_FILENAME} is valid ONNX model.")
    except onnx.onnx_cpp2py_export.checker.ValidationError as ex:
        print("{ONNX_FILENAME} model is not valid ONNX model: {}".format(ex))
    return

# Test the ONNX model on two images
def test_onnx():
    # Load and preprocess the first image
    img1 = Image.open("./n01440764_tench.jpeg")
    inp1 = mtailor_model.preprocess_numpy(img1).numpy().reshape(1, 3, 224, 224)

    # Load and preprocess the second image
    img2 = Image.open("./n01667114_mud_turtle.JPEG")
    inp2 = mtailor_model.preprocess_numpy(img2).numpy().reshape(1, 3, 224, 224)

    # Run the first image through the model
    out1 = session.run(None, {"input": inp1})[0]
    out1 = np.argmax(out1)

    # Run the second image through the model
    out2 = session.run(None, {"input": inp2})[0]
    out2 = np.argmax(out2)

    # Print the results
    print(f"Image 1: Predicted class id: {out1}")
    print(f"Image 2: Predicted class id: {out2}")

    # Check if the results are correct
    assert out1 == 0, f"Error: Image 1 should belong to class 0, but predicted {out1}"
    assert out2 == 35, f"Error: Image 2 should belong to class 35, but predicted {out2}"

# Run the test
if __name__ == '__main__':
    testing_validity_onnx_model()
    test_onnx()




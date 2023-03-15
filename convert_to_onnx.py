import torch
import onnx
import urllib.request
import io
import pytorch_model
from typing import Tuple

# Calling the defined model in the pytorch_model file
MODEL = pytorch_model.Classifier(pytorch_model.BasicBlock, [2, 2, 2, 2])

# Download the model weights from the link and load them into the PyTorch model
MODEL_WEIGHTS = "https://www.dropbox.com/s/b7641ryzmkceoc9/pytorch_model_weights.pth?dl=1"
INPUT_SHAPE = (3, 224, 224)
ONNX_FILENAME = "mtailor.onnx"


def converting_pytorch_model_to_onnx(model, model_weights: str, onnx_filename: str, input_shape: Tuple[float, float, float]):
    """Summary: Converts the pytorch model defined into a onnx model format
    Args:
        model (_type_): The defined pytorch model called from the pytorch_model file
        model_weights (str): the url to the stored model weights.
        onnx_filename (str): the output filename of the onnx model.
        input_shape (Tuple[float, float, float]): the input to the model. In our case, is an RGB image of size 224 by 224
    """
    model_weights_file = io.BytesIO(urllib.request.urlopen(model_weights).read())
    model.load_state_dict(torch.load(model_weights_file))
    dummy_input = torch.randn(1, *input_shape)
    torch.onnx.export(model, dummy_input, onnx_filename, export_params=True, opset_version=12, input_names=["input"], output_names=["output"])
    return


if __name__ == '__main__':
    converting_pytorch_model_to_onnx(MODEL, MODEL_WEIGHTS, ONNX_FILENAME, INPUT_SHAPE)



import torch
import onnx
import urllib.request
import io
import pytorch_model 


mtailor = pytorch_model.Classifier(pytorch_model.BasicBlock, [2, 2, 2, 2])

# Download the model weights from the link and load them into the PyTorch model
model_weights_url = "https://www.dropbox.com/s/b7641ryzmkceoc9/pytorch_model_weights.pth?dl=1"
print("weights urls", model_weights_url)
model_weights_file = io.BytesIO(urllib.request.urlopen(model_weights_url).read())
print("Model urls", model_weights_file)
mtailor.load_state_dict(torch.load(model_weights_file))

# Define the input shape of the model
input_shape = (3, 224, 224)
# Create a dummy input tensor for the model
dummy_input = torch.randn(1, *input_shape)

# Export the model to ONNX format
onnx_filename = "mtailor.onnx"

torch.onnx.export(mtailor, dummy_input, onnx_filename, export_params=True, opset_version=12, input_names=["input"], output_names=["output"])






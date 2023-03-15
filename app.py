from .model import OnnxModel


def init(model_name : str = "./models/mtailor.onnx"):
    global model 
    model = OnnxModel(model_name)

def inference(model_inputs: dict):
    global model
    # Parse out your arguments
    prompt = model_inputs.get("prompt", None)
    if prompt == None:
        return {"message": "No prompt provided"}

    pred = model.inference(prompt)
    return pred
**Deploy Classification Neural Network on Serverless GPU platform of Banana Dev**
This are the instructions for testing and deploying the models on Banana serverless framework.
## SET UP and Working on the Repository

1). Clone the github repository

$ git clone <repo name>

2) After cloning the github repository, set up the virtual environment by using the following commands

cd <repo name>
```
$ python -m venv .
$ source ./bin/activate
```

On windows cmd, run
```
Scripts\activate.bat
```

3). Install required dependencies
pip install -r requirements.txt

5). Convert the Pytorch model to onnx
python convert_to_onnx.py

6). Testing the server and the created onnx model
python tests/test_onnx.py
python tests/test_server.py

6). Run unit tests
pytest tests/

## To deploy the model
Every push to the main repository triggers the building of the model on Banana server and the existing model will be updated as required.
Basic inspiration for the deployment of the models can be obtained here (reference)[https://flyte.org/blog/how-to-serve-ml-models-with-banana]
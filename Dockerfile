# Must use a Cuda version 11+
#FROM python:3.9-slim
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

WORKDIR /

# Install git
RUN apt-get update && apt-get install -y git

# Install python packages
RUN pip3 install --upgrade pip
ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# We add the banana boilerplate here
ADD ./server.py .

# Add your model in onnx format
ADD ./models/mtailor.onnx ./


# Add your custom app code, init() and inference()
ADD  ./app.py .

EXPOSE 8000

CMD python3 -u server.py
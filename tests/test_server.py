import requests
import json
import time
import requests
import banana_dev as banana
import base64
from io import BytesIO

API_KEY = "22e1e370-5176-45ec-9562-24153f922bc8"
MODEL_KEY = "74355983-a06a-4bdc-92dd-684cd31c8a78"

def test_time_per_api_call():

    with open("./n01440764_tench.jpeg", "rb") as f:
        im_b64 = base64.b64encode(f.read()).decode("utf8")
    model_inputs = {
    "prompt": str(im_b64),
        }

    # Run the model
    start_time = time.time()
    out = banana.run(API_KEY, MODEL_KEY, model_inputs)
    end_time = time.time()
    time_taken = end_time - start_time
    print('Time taken:', time_taken , 'seconds')
    return time_taken
def test_new_image(image_path):

    with open(image_path, "rb") as f:
        im_b64 = base64.b64encode(f.read()).decode("utf8")

    model_inputs = {
    "prompt": str(im_b64),
        }

    # Run the model
    
    out = banana.run(API_KEY, MODEL_KEY, model_inputs)
    return out['class_id']

def baseline_test():
    test_images = ['./n01440764_tench.jpeg', './n01667114_mud_turtle.JPEG']
    expected_class_ids = [0, 35]

    for i, image_path in enumerate(test_images):
        with open(image_path, "rb") as f:
            im_bytes = f.read()
        # load the image
        im_b64 = base64.b64encode(im_bytes).decode("utf8")

        model_inputs = {'prompt', im_b64}
       
        output = banana.run(API_KEY, MODEL_KEY, model_inputs)
        assert output == expected_class_ids[i], f"Error: Image 1 should belong to class 0, but predicted {expected_class_ids[i]}"
       
if __name__ == '__main__':
    import argparse
    # define command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--preset', default='False', type=str,
                        help='path to save the ONNX model')
    args = parser.parse_args()

    image_file = 'n01440764_tench.jpeg'
    test_time_per_api_call()

    test_image(image_file)
    if args.preset:
        baseline_test()
    else:
        test_new_image(image_file)

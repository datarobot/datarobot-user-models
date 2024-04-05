"""
Usage:
    python datarobot-predict.py <input-file> [mimetype] [charset]
 
This example uses the requests library which you can install with:
    pip install requests
We highly recommend that you update SSL certificates with:
    pip install -U "urllib3[secure]" certifi
"""
import sys
from json import JSONDecoder

import numpy as np
from PIL import Image
from torchvision import transforms
import tritonclient.http as httpclient
import requests
import json

# See README.md on how to set up those keys
API_URL = '<DATAROBOT_API_URL>'
API_KEY = '<DATAROBOT_API_KEY>'
DATAROBOT_KEY = '<DATAROBOT_KEY>'
DEPLOYMENT_ID = '<DEPLOYMENT_ID>'

# Don't change this. It is enforced server-side too.
MAX_PREDICTION_FILE_SIZE_BYTES = 52428800  # 50 MB


class DataRobotPredictionError(Exception):
    """Raised if there are issues getting predictions from DataRobot"""


def make_datarobot_deployment_unstructured_predictions(data, deployment_id, mimetype, charset):
    """
    Make unstructured predictions on data provided using DataRobot deployment_id provided.
    See docs for details:
         https://docs.datarobot.com/en/docs/api/reference/predapi/dr-predapi.html
 
    Parameters
    ----------
    data : bytes
        Bytes data read from provided file.
    deployment_id : str
        The ID of the deployment to make predictions with.
    mimetype : str
        Mimetype describing data being sent.
        If mimetype starts with 'text/' or equal to 'application/json',
        data will be decoded with provided or default(UTF-8) charset
        and passed into the 'score_unstructured' hook implemented in custom.py provided with the model.
 
        In case of other mimetype values data is treated as binary and passed without decoding.
    charset : str
        Charset should match the contents of the file, if file is text.
 
    Returns
    -------
    data : bytes
        Arbitrary data returned by unstructured model.
 
 
    Raises
    ------
    DataRobotPredictionError if there are issues getting predictions from DataRobot
    """
    # Set HTTP headers. The charset should match the contents of the file.
    headers = {
        'Content-Type': '{};charset={}'.format(mimetype, charset),
        'Authorization': 'Bearer {}'.format(API_KEY),
        'DataRobot-Key': DATAROBOT_KEY,
    }

    url = API_URL.format(deployment_id=deployment_id)

    # Make API request for predictions
    predictions_response = requests.post(
        url,
        data=data,
        headers=headers,
    )
    _raise_dataroboterror_for_status(predictions_response)
    # Return raw response content
    return predictions_response.content


def _raise_dataroboterror_for_status(response):
    """Raise DataRobotPredictionError if the request fails along with the response returned"""
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError:
        err_msg = '{code} Error: {msg}'.format(
            code=response.status_code, msg=response.text)
        raise DataRobotPredictionError(err_msg)


def preprocess_image(img_path):
    img = Image.open(img_path)
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return np.expand_dims(preprocess(img).numpy(), axis=0)


def read_model_inference_header():
    with open('model_header.json') as f:
        json_header = json.load(f)
        json_header_str = json.dumps(json_header) + '\n'
    return json_header_str.encode('utf-8')


def binary_response_as_numpy(response):
    _, header_length = JSONDecoder().raw_decode(response.decode('utf-8'))
    results = httpclient.InferenceServerClient.parse_response_body(response, header_length=header_length)
    return results.as_numpy("fc6_1").astype(str)


def main(filename, deployment_id, mimetype, charset):
    """
    Return an exit code on script completion or error. Codes > 0 are errors to the shell.
    Also useful as a usage demonstration of
    `make_datarobot_deployment_unstructured_predictions(data, deployment_id, mimetype, charset)`
    """
    transformed_img = preprocess_image(filename)
    model_header = read_model_inference_header()

    # check the Triton Binary Tensor example, to get overview of the payload structure
    # https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_binary_data.md#examples
    data = model_header + transformed_img.tobytes()

    data_size = sys.getsizeof(data)
    if data_size >= MAX_PREDICTION_FILE_SIZE_BYTES:
        print((
                  'Input file is too large: {} bytes. '
                  'Max allowed size is: {} bytes.'
              ).format(data_size, MAX_PREDICTION_FILE_SIZE_BYTES))
        return 1
    try:
        response = make_datarobot_deployment_unstructured_predictions(data, deployment_id, mimetype, charset)
    except DataRobotPredictionError as exc:
        print(exc)
        return 1

    predictions = binary_response_as_numpy(response)
    print(np.squeeze(predictions)[:5])
    return 0


if __name__ == "__main__":
    filename = "img1.jpg"
    mimetype = "application/octet-stream"
    charset = "UTF-8"
    sys.exit(main(filename, DEPLOYMENT_ID, mimetype, charset))

import os
import grpc
import urllib.request
import numpy as np
import cv2

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


from flask import Flask
from flask import request
from flask import jsonify

from proto import np_to_protobuf

host = os.getenv('TF_SERVING_HOST', 'localhost:8500')

channel = grpc.insecure_channel(host)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)


def download_image(url):
    with urllib.request.urlopen(url) as url:
        img_array = np.asarray(bytearray(url.read()), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img

def preprocess(img_file, target_size=(256,256)):
    img_gray = cv2.cvtColor(img_file, cv2.COLOR_BGR2GRAY)
#     img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    img = img_gray / 255
    img = cv2.resize(img, target_size)
    img = np.reshape(img, img.shape + (1,))
    img = np.reshape(img,(1,) + img.shape)
    return img


def prepare_request(X):
    pb_request = predict_pb2.PredictRequest()

    pb_request.model_spec.name = 'unet_lung_model'
    pb_request.model_spec.signature_name = 'serving_default'

    pb_request.inputs['input_1'].CopyFrom(np_to_protobuf(X.astype(np.float32)))
    return pb_request

def prepare_response(pb_response):
    preds = pb_response.outputs['conv2d_18'].float_val
    pred_arr = np.array(list(preds))
    pred_arr = np.reshape(pred_arr, (512, 512))
    pred_arr = (pred_arr * 255.).astype(np.uint8)
    return pred_arr

def add_colored_mask(image, mask_image):
    mask_image_color = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)
    mask = cv2.bitwise_and(mask_image_color, mask_image_color, mask=mask_image)
    mask_coord = np.where(mask!=[0,0,0])
    mask[mask_coord[0],mask_coord[1],:]=[255,0,0]
    ret = cv2.addWeighted(image, 0.9, mask, 0.3, 0)
    return ret

def predict(url):
    img = download_image(url)
    X = preprocess(img, target_size=(512,512))
    pb_request = prepare_request(X.astype(np.float32))
    pb_response = stub.Predict(pb_request, timeout=20.0)
    response = prepare_response(pb_response)
    result = add_colored_mask(img, response)
    return result

app = Flask('gateway')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    data = request.get_json()
    url = data['url']
    result = predict(url)
    result = result.tolist()
    return jsonify(result)


if __name__ == '__main__':
    # url = 'https://raw.githubusercontent.com/rahmatmamat1/unet_lung_segmentation/main/chest_xray.png'
    # response = predict(url)
    # print(type(response))
    app.run(debug=True, host='0.0.0.0', port=9696)
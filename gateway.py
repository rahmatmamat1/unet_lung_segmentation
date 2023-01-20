#!/usr/bin/env python
# coding: utf-8

import os
import grpc

import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from keras_image_helper import create_preprocessor

from flask import Flask
from flask import request
from flask import jsonify

from proto import np_to_protobuf

host = os.getenv('TF_SERVING_HOST', 'localhost:8500')

channel = grpc.insecure_channel(host)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

preprocessor = create_preprocessor('xception', target_size=(299, 299))


def prepare_request(X):
    pb_request = predict_pb2.PredictRequest()

    pb_request.model_spec.name = 'unet_lung_model'
    pb_request.model_spec.signature_name = 'serving_default'

    pb_request.inputs['input_1'].CopyFrom(np_to_protobuf(X))
    return pb_request

def prepare_response(pb_response):
    preds = pb_response.outputs['conv2d_18'].float_val
    pred_arr = np.array(list(preds))
    pred_arr = np.reshape(pred_arr, (512, 512))
    pred_arr = (pred_arr * 255.).astype(np.uint8)
    return pred_arr


def predict(url):
    X = preprocessor.from_url(url)
    pb_request = prepare_request(X)
    pb_response = stub.Predict(pb_request, timeout=20.0)
    response = prepare_response(pb_response)
    return response

def visualize(image, pred_arr):
    fig, axs = plt.subplots(1, 2, figsize=(10, 10))
    image_test = cv2.resize(image, (512,512))

    axs[0].set_title("Image")
    axs[0].imshow(image)
    axs[1].set_title("Mask")
    axs[1].imshow(add_colored_mask(image_test, pred_arr))
    plt.show()

app = Flask('gateway')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    data = request.get_json()
    url = data['url']
    result = predict(url)
    return jsonify(result)


if __name__ == '__main__':
    # url = 'http://bit.ly/mlbookcamp-pants'
    # response = predict(url)
    # print(response)
    app.run(debug=True, host='0.0.0.0', port=9696)
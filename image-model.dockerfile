FROM tensorflow/serving:2.7.0

COPY unet_lung_model /models/unet_lung_model/1
ENV MODEL_NAME="unet_lung_model"
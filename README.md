# Project Title: Lung Segmentation on Chest X-Ray Images using U-Net Model
## Problem Statement
Chest X-ray is a commonly used diagnostic tool for lung diseases. However, manual annotation of lung regions in chest X-rays is a time-consuming and tedious task. The goal of this project is to develop a deep learning model for automatic lung segmentation on chest X-ray images, which can aid radiologists in their diagnostic workflow.

## Dataset
This project uses datasets collected by the National Library of Medicine, Maryland, USA, in collaboration with Shenzhen No.3 People's Hospital, Guangdong Medical College, Shenzhen, China. Chest X-Ray originates from an outpatient clinic and is part of the daily routine using the Philips DR Digital Diagnosis system.

While the segmentation mask data was created manually by students and teachers from the Computer Engineering Department, Faculty of Informatics and Computer Engineering, National Technical University of Ukraine "Igor Sikorsky Kyiv Polytechnic Institute," Kyiv, Ukraine.

Data can be seen at the following link.
https://www.kaggle.com/datasets/kmader/pulmonary-chest-xray-abnormalities
https://www.kaggle.com/datasets/yoctoman/shcxr-lung-mask

## Methodology
The U-Net architecture was chosen for this project due to its success in image segmentation tasks. The model was trained on a dataset of chest X-ray images and corresponding lung masks. The performance was evaluated using metrics such as Dice coefficient and Jaccard index.

## Results
The model was able to achieve a Dice coefficient of 0.89 and Jaccard index of 0.81 on the test set.

## Deployment
The model was deployed using Flask and TensorFlow Serving. Flask was used to create a web service, while TensorFlow Serving was used to serve the model as a API.

## Files
* `Pipfile` and `Pipfile.lock` : library requirement for deployment
* `u-net-lung-segmentation.ipynb` : Notebook file contains EDA and Modeling, originally from kaggle [notebook](https://www.kaggle.com/code/rahmatsyahfirdaus/u-net-lung-segmentation/notebook)
* `save-test-model.ipynb` : Notebook file to save and test TensorFlow Serving
* `gateway.py` : python script for serving ML models using Flask
* `proto.py` : python script for convert np tp protobuf
* `test.py` : python script to send a request to the API
* `image-gateway.dockerfile` : Docker file build service container
* `image-model.dockerfile` : Docker file to build model server container
* `docker-compose.yaml` : to run service and model server container

## How to Run
1. Clone the repository
    ```bash
    git clone https://github.com/rahmatmamat1/unet_lung_segmentation.git
    ```
2. Download and save model

    If you already install Kaggle API, run this:
    ```bash
    kaggle kernels output rahmatsyahfirdaus/u-net-lung-segmentation -p /path/to/dest
    ```
    or you can just download it at original kaggle notebook [here](https://www.kaggle.com/code/rahmatsyahfirdaus/u-net-lung-segmentation/notebook).

    Run `save-test-model.ipynb` to save model so it can be serve using tensorflow serving.
3. Build docker image for service and model server

    Docker image model server
    ```bash
    docker build -t lung-seg-model:unet-001 -f image-model.dockerfile .
    ```
    Docker image service
    ```bash
    docker build -t lung-seg-gateway:001 -f image-gateway.dockerfile .
    ```
4. Containerize and run both image locally.

    We've created our service and model server image, now let's dockerize and run both.
    ```bash
    docker-compose up -d
    ```
5. Test the API

    Use the API by sending a POST request to http://localhost:9696/predict with the input image in the request body. you can simply run `test.py` file.
    ```bash
    python test.py
    ```




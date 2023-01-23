import requests
import numpy as np
import matplotlib.pyplot as plt

url = 'http://localhost:9696/predict'

data = {'url': 'https://raw.githubusercontent.com/rahmatmamat1/unet_lung_segmentation/main/chest_xray.png'}

result = requests.post(url, json=data).json()
result_arr = np.array(result)
plt.imshow(result_arr)
plt.show()
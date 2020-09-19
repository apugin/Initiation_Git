from skimage import io
from skimage import transform
import matplotlib.pyplot as plt
import numpy as np


def data_set(**kwargs):  # fonctoin à utiliser dans test et train
    return np.random.random((16,32,32,3))


def modif():  # instructions pour modif
    image = io.imread("C:/Users/lucas/Desktop/Chris_AI/Projet/data/Kvasir-SEG/images/cju0qkwl35piu0993l0dewei2.jpg")
    resized_image = transform.resize(image, (32, 32))
    io.imshow(resized_image)
    plt.show()






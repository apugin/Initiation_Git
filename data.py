from skimage import io
from skimage import transform
from skimage import metrics
import os
import tensorflow as tf
import numpy as np

def false_data_set(**kwargs):  # fonctoin à utiliser dans test et train
    return np.random.random((16,32,32,3))


def data_set_32(pictures_tab):  # instructions pour modif, en entrée un tableau d'images, en sortie le tableau d'images modifiées
    n = len(pictures_tab)
    resized_pictures_tab = []
    for k in range(n):
        resized_picture = modif_picture_32(pictures_tab[k])
        resized_pictures_tab.append(resized_picture)
    return resized_pictures_tab

def modif_picture_32(picture):  #instructions pour modifier les images une par une
    resized_picture = transform.resize(picture, (32, 32))
    return resized_picture


def data_set_64(pictures_tab):  # instructions pour modif, en entrée un tableau d'images, en sortie le tableau d'images modifiées
    n = len(pictures_tab)
    resized_pictures_tab = []
    for k in range(n):
        resized_picture = modif_picture_64(pictures_tab[k])
        resized_pictures_tab.append(resized_picture)
    return resized_pictures_tab

def modif_picture_64(picture):  #instructions pour modifier les images une par une
    resized_picture = transform.resize(picture, (64, 64))
    return resized_picture


dirpath = "C:/Users/lucas/Desktop/Chris_AI/Projet/data/Kvasir-SEG/images"


def create_pictures_tab(dirpath):
    list_files = os.listdir(dirpath)
    n = len(list_files)
    all_picture_tab = []
    for k in range(n):
        final_path = dirpath + "/" + list_files[k]
        picture = io.imread(final_path)
        all_picture_tab.append(picture)
    return all_picture_tab

data_set = create_pictures_tab(dirpath)

data_set__32 = data_set_32(data_set)

data_set__64 = data_set_64(data_set)

data_set_32_64 = data_set_64(data_set__64)

def variance(false_data_set,true_data_set):
    Metric = []
    n = len(false_data_set)
    for k in range(n):
        m = metrics.peak_signal_noise_ratio(true_data_set[k], false_data_set[k], data_range=None)
        Metric.append(m)
    mean = sum(Metric)/n
    variance_2 = 0
    for k in range(n):
        variance_2 = variance_2 + (Metric[k] - mean)/n
    variance = variance_2**(1/2)
    return (variance)

def len_test_data_set(false_data_set,true_data_set,alpha):  # pour un alpha donne, on veut une précision d'au moins 1-alpha
    var = variance(false_data_set,true_data_set)
    n_min = var/(alpha**3)
    return n_min

len_test = len_test_data_set(data_set__64, data_set_32_64,0.05)

print(len_test)




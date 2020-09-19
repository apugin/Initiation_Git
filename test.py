from data import data_set
from model import build_network
from utilities import *


def test(epoch, ckpt_dir):
    network = build_network()
    network.load_weights(ckpt_dir)  # Comment récupérer les poids d'une epoch donnée ?
    data_test = data_set('test')
    inputs = data_test[0]
    label = data_test[1]
    predicted = network.predict(inputs)
    PSNR = psnr(label, predicted)
    SSIM = ssim(label, predicted)
    print('PSNR :')
    print(PSNR)
    print('SSIM :')
    print(SSIM)

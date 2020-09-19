import numpy as np
import tensorflow as tf
import argparse
import os
# from model import super_resolution
from test import test
from train import train

parser = argparse.ArgumentParser(description='')
parser.add_argument('--epoch', dest='epoch', type=int, default=10, help='number of epochs')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=128, help='number of images in batch')
parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='initial learning rate for adam')
# parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--phase', dest='phase', default='train', help='train or test')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./checkpoint', help='models are saved here')
# Il faut décider de où on stocke les poids
parser.add_argument('--test_dir', dest='test_dir', default='./data/test', help='denoised sample are saved here')
# Il faut décider de où on stocke les images dont on a baissé la résolution
args = parser.parse_args()

# Quel nom donner au réseau ? Je pars sur super_resolution
# Forme des fonctions :
# build_network(weights=None) [ne pas le faire dans main]
# train(epoch, batch_size, lr)
# test(ckpt_dir, test_dir)


def main():
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)

    lr = args.lr * np.ones([args.epoch])
    # Pour changer la valeur du learning rate selon l'epoch

    if args.phase == 'train':
        train(args.epoch, args.batch_size, lr)

    elif args.phase == 'test':
        test(args.ckpt_dir, args.test_dir)

    else :
        print("/!\ Unknown phase : type 'train' or 'test'")
        exit(0)


if __name__ == '__main__':
    main()
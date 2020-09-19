import numpy as np
import argparse
import os
from test import test
from train import train

parser = argparse.ArgumentParser(description='')
parser.add_argument('--epoch', dest='epoch', type=int, default=10, help='number of epochs')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=128, help='number of images in batch')
parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='initial learning rate for adam')
# parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--phase', dest='phase', default='train', help='train or test')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./checkpoints', help='models are saved here')
# Il faut décider de où on stocke les poids
args = parser.parse_args()


def main():
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    lr = args.lr * np.ones([args.epoch])
    # Pour changer la valeur du learning rate selon l'epoch

    if args.phase == 'train':
        train(batchsize=args.batch_size, epochs=args.epoch, lr=lr)

    elif args.phase == 'test':
        test(epoch=args.epoch, ckpt_dir=args.ckpt_dir)

    else:
        print("/!\ Unknown phase : type 'train' or 'test'")
        exit(0)


if __name__ == '__main__':
    main()

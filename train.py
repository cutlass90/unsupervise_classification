import os
import argparse

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

from vae import VAE

input_dim = 28*28
save_path = 'models/mnist_dnn/m'
os.makedirs('models/mnist_dnn', exist_ok=True)

parser = argparse.ArgumentParser(
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
                    '--z_dim',  type=int,
                        default=12, help='latent code size')
parser.add_argument(
                    '--n_clusters',  type=int,
                        default=10, help='number of clusters')
parser.add_argument(
                    '--batch_size',  type=int,
                        default=64, help='batch size')
parser.add_argument(
                    '--num_epochs', type=int,
                        default=200, help='maximum number of training epochs')
parser.add_argument(
                    '--learning_rate', type=float,
                        default=0.001, help='learning rate')

load_parser = parser.add_mutually_exclusive_group(required=False)
load_parser.add_argument('--restore', dest='restore', action='store_true')
parser.set_defaults(restore=False)

args = parser.parse_args()

model = VAE(input_dim=input_dim,
    n_clusters=args.n_clusters,
    z_dim = args.z_dim)
if args.restore:
    model.load_model(save_path)

for e in range(1, args.num_epochs+1):
	print('\n', '-'*30, 'Epoch {}'.format(e), '-'*30, '\n')
	model.train(args.batch_size, args.learning_rate, mnist)
	model.predict(mnist)
	model.save_model(save_path)

os.system('CUDA_VISIBLE_DEVICES=1 python3 plot.py')

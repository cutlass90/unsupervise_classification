import os

from tensorflow.examples.tutorials.mnist import input_data

from vae_parameters import parameters as PARAM
import vae_tools as tools
from vae import VAE

save_path = 'models/mnist_dnn/m'
os.makedirs('models/mnist_dnn', exist_ok=True)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

model = VAE(input_dim=PARAM['input_dim'],
    n_clusters=PARAM['n_clusters'],
    z_dim=PARAM['z_dim'])
if PARAM['restore']:
    model.load_model(save_path)

for e in range(1, PARAM['n_epochs']+1):
	print('\n', '-'*30, 'Epoch {}'.format(e), '-'*30, '\n')
	model.train(PARAM['batch_size'], PARAM['learn_rate'], mnist)
	model.predict(mnist)
	model.save_model(save_path)

os.system('CUDA_VISIBLE_DEVICES=1 python3 plot.py')

tools.plot_clusters(model=model, n_clusters=PARAM['n_clusters'],
    z_dim=PARAM['z_dim'])
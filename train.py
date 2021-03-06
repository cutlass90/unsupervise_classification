import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from vae_parameters import parameters as PARAM
import vae_tools as tools
from vae import VAE

save_path = 'models/mnist_dnn/m'
os.makedirs('models/mnist_dnn', exist_ok=True)


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# train
with VAE(
    input_dim=PARAM['input_dim'],
    n_clusters=PARAM['n_clusters'],
    z_dim=PARAM['z_dim'],
    sampling=PARAM['sampling']) as model:

    if PARAM['restore']:
        model.load_model(save_path)

    
    for e in range(1, PARAM['n_epochs']+1):
        print('\n', '-'*30, 'Epoch {}'.format(e), '-'*30, '\n')
        model.train(batch_size=PARAM['batch_size'],
            learning_rate=PARAM['learn_rate'],
            data_loader=mnist,
            KL_weight=PARAM['KL_weight'],
            entropy_weight=PARAM['entropy_weight'],
            m_class_weight=PARAM['m_class_weight'])
        model.predict(data_loader=mnist, KL_weight=PARAM['KL_weight'],
            entropy_weight=PARAM['entropy_weight'],
            m_class_weight=PARAM['m_class_weight'])
        model.save_model(save_path)
    


# test
with VAE(
    input_dim=PARAM['input_dim'],
    n_clusters=PARAM['n_clusters'],
    z_dim=PARAM['z_dim'],
    sampling=False) as model:

    model.load_model(save_path)

    tools.view_z(model)
    tools.plot_clusters_latent_space(model=model, n_clusters=PARAM['n_clusters'])
    tools.plot_clusters(model=model, n_clusters=PARAM['n_clusters'],
        z_dim=PARAM['z_dim'])


    tools.reconstruct(model, n_clusters=10, k=5, n_images=10)
    tools.calc_acc(model=model,
    	n_clusters=PARAM['n_clusters'])

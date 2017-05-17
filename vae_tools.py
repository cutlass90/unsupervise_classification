import random
import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


def to_image(x):
    return x.reshape(28, 28)

def reconstruct(k=4):
    original = random.sample(list(mnist.validation.images), k)
    reconstruction = list(model.reconstruct(original))

    fig = plt.figure(figsize=(4, k*2))
    for orig, rec, i in zip(original, reconstruction, range(k)):
        ax = fig.add_subplot(k, 2, 2*i+1)
        ax.imshow(to_image(orig), cmap='gray')
        ax.set_title('original')
        ax.set_axis_off()
        ax = fig.add_subplot(k, 2, 2*i+2)
        ax.imshow(to_image(rec), cmap='gray')
        ax.set_title('reconstructed')
        ax.set_axis_off()
    os.makedirs('pics', exist_ok=True)
    fig.savefig('pics/reconstruction.png')
    plt.close()
    print('Reconstruction saved.')

def sample():

    fig = plt.figure(figsize=(7, 7))
    for i in range(225):
        ax = fig.add_subplot(15, 15, i+1)
        z = np.random.normal(size=[1, z_dim]).astype(np.float32)
        x = model.reconstruct_from_z(z)
        ax.imshow(to_image(x), cmap='gray', aspect='auto')
        ax.set_axis_off()

    #remove spacings between subplots
    fig.subplots_adjust(hspace=0, wspace=0, left=0, bottom=0, right=1, top=1)
    os.makedirs('pics', exist_ok=True)
    fig.savefig('pics/sample_from_latent_space.png')
    plt.close()
    print('Sample saved.')

def plot_latent_space():
    nx = ny = 20
    x_values = np.linspace(-3, 3, nx)
    y_values = np.linspace(-3, 3, ny)

    canvas = np.empty((28*ny, 28*nx))
    for i, yi in enumerate(x_values):
        for j, xi in enumerate(y_values):
            z_mu = [[xi, yi]]
            x_mean = model.reconstruct_from_z(z_mu)
            canvas[(nx-i-1)*28:(nx-i)*28, j*28:(j+1)*28] = to_image(x_mean[0])

    fig = plt.figure(figsize=(5, 5))        
    Xi, Yi = np.meshgrid(x_values, y_values)
    ax = fig.add_subplot(111)
    ax.imshow(canvas,  origin='upper', cmap='gray', interpolation='bicubic')
    ax.set_axis_off()
    os.makedirs('pics', exist_ok=True)
    fig.savefig('pics/latent_space.png')
    plt.close()
    print('Latent space saved.')

def plot_clusters_latent_space(model, n_clusters):
    for cl in range(n_clusters):
        nx = ny = 20
        x_values = np.linspace(-2, 2, nx)
        y_values = np.linspace(-2, 2, ny)

        canvas = np.empty((28*ny, 28*nx))
        z_arr = np.empty([0, n_clusters+2], dtype=np.float32)
        for i, yi in enumerate(x_values):
            for j, xi in enumerate(y_values):
                z = np.zeros([1, n_clusters+2])
                z[0, n_clusters:] = [xi, yi]
                z[0, cl] = 0.99
                z = z.astype(np.float32)
                z_arr = np.vstack([z_arr,z])
        x_mean = model.reconstruct_from_z(z_arr)
        for i, yi in enumerate(x_values):
            for j, xi in enumerate(y_values):
                canvas[(nx-i-1)*28:(nx-i)*28, j*28:(j+1)*28] = to_image(x_mean[nx*i+j,:])

        fig = plt.figure(figsize=(7, 7))        
        Xi, Yi = np.meshgrid(x_values, y_values)
        ax = fig.add_subplot(111)
        ax.imshow(canvas,  origin='upper', cmap='gray', interpolation='bicubic')
        ax.set_axis_off()
        os.makedirs('pics', exist_ok=True)
        fig.savefig('pics/cluster_{}_latent_space.png'.format(cl))
        print('Latent space of cluster {} was saved.'.format(cl))
        plt.close()


def plot_clusters(model, n_clusters, z_dim):
    n_exs = 15

    z_arr = np.empty([0, n_clusters+2], dtype=np.float32)
    for i in range(n_exs):
        z_mu = np.random.normal(size=[z_dim-n_clusters]).astype(np.float32)
        for c in range(n_clusters):
            im_n = i*n_clusters+c+1
            z = np.zeros([1,z_dim]).astype(np.float32)
            z[0, n_clusters:] = z_mu
            z[0, c] = 0.99
            z_arr = np.vstack([z_arr,z])
  
    x = model.reconstruct_from_z(z_arr)

    fig = plt.figure(figsize=(7, 7))
    for i in range(n_exs):
        for c in range(n_clusters):
            im_n = i*n_clusters+c+1
            ax = fig.add_subplot(n_exs, n_clusters, im_n)
            ax.imshow(to_image(x[im_n-1,:]), cmap='gray', aspect='auto')
            ax.set_axis_off()
    #remove spacings between subplots
    fig.subplots_adjust(hspace=0, wspace=0, left=0, bottom=0, right=1, top=1)
    os.makedirs('pics', exist_ok=True)
    fig.savefig('pics/plot_clusters.png')
    plt.close()
    print('plot_clusters saved.')

def view_z(model, n_samles=10):
    x, y = mnist.train.next_batch(n_samles)
    z = model.get_z(x)
    plt.figure(figsize=[15,15])
    for i in range(n_samles):
        plt.subplot(n_samles, 1, i+1)
        plt.plot(z[i], label=str(y[i]))
        plt.legend()
    plt.savefig('pics/z_code.png')
    plt.close()
    print('z code saved')


def calc_acc(model, n_clusters):
    # label_map is a dict where key is  number of cluster, starting from 0
        # and value is true label
    batch_size = 64
    label_map = get_label_map(model, n_clusters, batch_size)
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    acc_list = []
    for i in range(mnist.train.num_examples//batch_size):
        x, y = mnist.train.next_batch(batch_size)
        clusters = model.get_z(x)[:, :n_clusters]
        y_pred = np.argmax(clusters, axis=1)
        # print('y_pred', y_pred)
        y_pred = np.array([label_map[cl] for cl in y_pred])
        acc_list.append((y==y_pred).sum()/batch_size)
    print('Accuracy = ', np.array(acc_list).mean())


def get_label_map(model, n_clusters, batch_size):
    x, y = mnist.train.next_batch(batch_size)
    clusters = model.get_z(x)[:, :n_clusters]
    clusters = (clusters>0.5).astype(float)
    label_map = map_labels_to_clusters(y, clusters)
    return label_map


def map_labels_to_clusters(labels, clusters):
    # label and clusters are matrices with one-hot vectors
    label_map = {i:identify_label(labels, clusters[:,i])
        for i in range(clusters.shape[1])}
    assert len(label_map)==clusters.shape[1], 'Can not гnequivocally identify all cluster'
    print('label_map', label_map)
    return label_map
    

def identify_label(labels, cluster):
    """ return label of specific vector """
    cluster = np.array(cluster)
    l = []
    for i in range(labels.shape[1]):
        a = np.sum(cluster==labels[:,i])
        l.append(a)
    return np.argmax(l,0)


    





if __name__ == '__main__':
    # testing

    # identify_label
    labels = np.eye(5)
    cluster = np.array([0,1,0,0,0])
    print('labels\n',labels)
    print('cluster\n', cluster)
    l = identify_label(labels, cluster)
    assert l == 1, 'test identify_label not passed'
    print(l)
    


    # sample()
    # plot_latent_space()
    

from __future__ import absolute_import, division, print_function, unicode_literals

import uuid

import tensorflow as tf
import PIL
import os
import numpy as np
from PIL import Image

from cleverhans.attacks import FastGradientMethod
from cleverhans.loss import CrossEntropy
from cleverhans.utils_tf import model_eval, train

from IterativeModel import IterativeModel

def parse_disk_file(data_dir):
    assert os.path.exists(data_dir), data_dir
    filenames = [filename for filename in os.listdir(data_dir) if filename.endswith('.png')]
    dim_sizes = (len(filenames), 28, 28, 1)
    y = np.zeros(shape=(len(filenames), 10))
    x = np.zeros(shape=dim_sizes)
    for index in range(len(filenames)):
        filename = filenames[index]
        label = int(filename.split('_')[0])
        y[index][label] = 1.0
        raw_image = np.asarray(PIL.Image.open(data_dir + '/' + filename)).reshape((28, 28, 1))
        x[index] = raw_image
    return x, y


def get_mnist(data_dir, train_start=0,
              train_end=600, test_start=0, test_end=100):
    assert isinstance(train_start, int)
    assert isinstance(train_end, int)
    assert isinstance(test_start, int)
    assert isinstance(test_end, int)

    X_train, Y_train = parse_disk_file('data/' + data_dir + '/train')
    X_test, Y_test = parse_disk_file('data/' + data_dir + '/test')
    return X_train, Y_train, X_test, Y_test


BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001
NB_FILTERS = 64
NB_CLASSES = 10

# Session
config_args = {}
sess = tf.Session(config=tf.ConfigProto(**config_args))

# New model
#model1 = IterativeModel('model1', NB_CLASSES, NB_FILTERS, 'pure', 'model1_adv')
#model1.checkpoint_path = 'model/NN_0/cp.ckpt'

# Start training
def train_model(model, save_model=False, generate_adv=False):
    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))
    # Load image from disk
    x_train, y_train, x_test, y_test = get_mnist(model.input_dir)
    # Train an MNIST model
    train_params = {
        'nb_epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE
    }
    eval_params = {'batch_size': BATCH_SIZE}
    rng = np.random.RandomState([2017, 8, 30])
    def do_eval(preds, x_set, y_set):
        acc = model_eval(sess, x, y, preds, x_set, y_set, args=eval_params)
        print('Test accuracy on train: %0.4f' % (acc))

    preds = model.get_logits(x)
    loss = CrossEntropy(model, smoothing=0.1)

    def evaluate():
        do_eval(preds, x_test, y_test)

    train(sess, loss, x, y, x_train, y_train, evaluate=evaluate, args=train_params, rng=rng, var_list=model.get_params())
    # Calculate training error
    do_eval(preds, x_train, y_train)
    if save_model:
        print('Saving model to: ' + model.checkpoint_path)
        model.save_weights(model.checkpoint_path)

    if generate_adv:
        x_train_pure, y_train_pure, x_test_pure, y_test_pure = get_mnist('pure')
        dir = model.output_dir
        fgsm = FastGradientMethod(model)
        fgsm_params = {
            'eps': 0.3,
            'clip_min': 0.,
            'clip_max': 1.
        }
        if not os.path.exists(dir + '/train/'):
            os.mkdir(dir + '/train/')
        if not os.path.exists(dir + '/test/'):
            os.mkdir(dir + '/test/')
        for index in range(len(y_test)):
            print('test ' + str(index))
            x_ = x_test_pure[index]
            label = np.argmax(y_test[index])
            raw_data = (fgsm.generate_np(x_.reshape((1, 28, 28, 1)), **fgsm_params).reshape((28, 28)) * 255).astype(
                'uint8')
            im = Image.fromarray(raw_data, mode='P')
            im.save(dir + 'test/' + str(label) + '_' + str(uuid.uuid4()) + '.png')
        for index in range(len(y_train)):
            print('train ' + str(index))
            x_ = x_train_pure[index]
            label = np.argmax(y_train[index])
            raw_data = (fgsm.generate_np(x_.reshape((1, 28, 28, 1)), **fgsm_params).reshape((28, 28)) * 255).astype(
                'uint8')
            im = Image.fromarray(raw_data, mode='P')
            im.save(dir + 'train/' + str(label) + '_' + str(uuid.uuid4()) + '.png')

model1 = IterativeModel('model1', NB_CLASSES, NB_FILTERS, 'pure', 'model1_adv')
model1.checkpoint_path = 'model/NN_0/cp.ckpt'
train_model(model1)

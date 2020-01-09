from __future__ import absolute_import, division, print_function, unicode_literals

import uuid

import tensorflow as tf
import PIL
import os
import numpy as np
from PIL import Image

from cleverhans.attacks import FastGradientMethod, optimize_linear
from cleverhans.compat import softmax_cross_entropy_with_logits, reduce_sum, reduce_max
from cleverhans.loss import CrossEntropy
from cleverhans.utils_tf import model_eval
from cleverhans.train import train
from IterativeModel import ModelBasicCNN


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
    return x.astype(np.float32), y.astype(np.float32)


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
config_args = {'allow_soft_placement' : True}
sess = tf.Session(config=tf.ConfigProto(**config_args))

# Define input TF placeholder
x = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 28, 28, 1))
y = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 10))

# Training function
def train_model(model):
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

    train(sess, loss, x_train, y_train, evaluate=evaluate, args=train_params, rng=rng)
    # Calculate training error
    do_eval(preds, x_train, y_train)


def eval_model(model, input_dir):
    # Load image from disk
    _, _, x_test, y_test = get_mnist(input_dir)
    preds = model.get_logits(x)
    loss = CrossEntropy(model, smoothing=0.1)
    # Evaluate an MNIST model
    eval_params = {'batch_size': BATCH_SIZE}
    acc = model_eval(sess, x, y, preds, x_test, y_test, args=eval_params)
    print('Test accuracy on train: %0.4f' % (acc))


def save_model(model):
    saver = tf.train.Saver()
    print(model.checkpoint_path)
    saver.save(sess, model.checkpoint_path)
    print("Model saved to: {}".format(model.checkpoint_path))


def load_model(path):
    saver = tf.train.Saver()
    print(path)
    saver.restore(sess, path)
    print("Model loaded from: {}".format(path))

def generate_adv_model(model):
    x_train_pure, y_train_pure, x_test_pure, y_test_pure = get_mnist('pure')
    preds = model.get_logits(x)
    dir = model.output_dir
    fgsm = FastGradientMethod(model, sess=sess)
    fgsm_params = {
        'eps': 0.3,
        'clip_min': 0.,
        'clip_max': 1.
    }
    dir = 'data/' + dir
    if not os.path.exists(dir + '/train/'):
        os.mkdir(dir + '/train/')
    if not os.path.exists(dir + '/test/'):
        os.mkdir(dir + '/test/')
    start = 0
    end = start + 128
    for index in range(len(y_test_pure) // 128):
        print('test ' + str(index))
        print('train ' + str(index))
        x_ = x_test_pure[start:end] / 255
        raw_data = (fgsm.generate_np(x_, **fgsm_params) * 255).reshape((128, 28, 28)).astype('uint8')
        for slic in range(start, end):
            if slic==len(y_test_pure):
                break
            im = Image.fromarray(raw_data[slic-start], mode='P')
            label = np.argmax(y_test_pure[slic])
            im.save(dir + '/test/' + str(label) + '_' + str(uuid.uuid4()) + '.png')
        start += 128
        end += 128

    start = 0
    end = start + 128
    for index in range(len(y_train_pure) // 128):
        print('train ' + str(index))
        print(start, end)
        x_ = x_train_pure[start:end] / 255
        raw_data = (fgsm.generate_np(x_, **fgsm_params) * 255).reshape((128, 28, 28)).astype('uint8')
        for slic in range(start, end):
            if slic==len(y_train_pure):
                break
            im = Image.fromarray(raw_data[slic-start], mode='P')
            label = np.argmax(y_train_pure[slic])
            im.save(dir + '/train/' + str(label) + '_' + str(uuid.uuid4()) + '.png')
        start += 128
        end += 128


# New model
model = ModelBasicCNN('model1', NB_CLASSES, NB_FILTERS)
model.checkpoint_path = 'model/NN_0/cp.ckpt'
model.input_dir = 'pure'
model.output_dir = 'model1_adv'
# train_model(model)
# save_model(model)
load_model('model/NN_0/cp.ckpt')
generate_adv_model(model)
# eval_model(model, 'pure')

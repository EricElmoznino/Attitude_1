import tensorflow as tf
import numpy as np
import time
import os
import webbrowser
from subprocess import Popen, PIPE


class Configuration:
    def __init__(self, train_log_path = './train', epochs=1, batch_size=50, dropout=0.5):
        self.train_log_path = train_log_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.keep_prob = 1-dropout


def data_at_path(path):
    files = os.listdir(path)
    files = sorted(files, key=lambda file: int(file.split('_')[0]))
    files = list(filter(lambda file: file.split('_')[1] == 'new', files))
    files_l = list(filter(lambda file: file.split('_')[-1] == 'l.jpg', files))
    files_r = list(filter(lambda file: file.split('_')[-1] == 'r.jpg', files))
    attitude_strings = [file.split('_')[2] for file in files_l]
    attitudes = [[float(s.split('x')[0]), float(s.split('x')[1]), float(s.split('x')[2])]
                 for s in attitude_strings]
    files_l = [os.path.join(path, f) for f in files_l]
    files_r = [os.path.join(path, f) for f in files_r]
    return files_l, files_r, attitudes


def log_step(step, total_steps, start_time, angle_error):
    progress = int(step / float(total_steps) * 100)

    seconds = time.time() - start_time
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    print(str(progress) + '%\t|\t',
          int(h), 'hours,', int(m), 'minutes,', int(s), 'seconds\t|\t',
          'Step:', step, '/', total_steps, '\t|\t',
          'Average Angle Error (Degrees):', angle_error)


def log_epoch(epoch, total_epochs, angle_error):
    print('\nEpoch', epoch, 'completed out of', total_epochs,
          '\t|\tAverage Angle Error (Degrees):', angle_error)


def log_generic(angle_error, set_name):
    print('Average Angle Error (Degrees) on', set_name, 'set:', angle_error, '\n')


def weight_variables(shape, mean=0.1):
    initial = tf.truncated_normal_initializer(stddev=0.1, mean=mean)
    return tf.get_variable('weights', shape=shape,
                           initializer=initial)


def bias_variables(shape):
    initial = tf.constant_initializer(0.1)
    return tf.get_variable('biases', shape=shape,
                           initializer=initial)


def convolve(model, window, n_inputs, n_outputs, stride=None, pad=False):
    if pad: padding = 'SAME'
    else: padding = 'VALID'
    with tf.variable_scope('convolution'):
        if stride is None: stride = [1, 1]
        weights = weight_variables(window + [n_inputs] + [n_outputs])
        biases = bias_variables([n_outputs])
        stride = [1] + stride + [1]
        return tf.nn.conv2d(model, weights, stride, padding=padding) + biases


def max_pool(model, pool_size, stride=None, pad=False):
    if pad: padding = 'SAME'
    else: padding = 'VALID'
    if stride is None: stride = [1] + pool_size + [1]
    else: stride = [1] + stride + [1]
    pool_size = [1] + pool_size + [1]
    return tf.nn.max_pool(model, ksize=pool_size, strides=stride, padding=padding)


def open_tensorboard():
    tensorboard = Popen(['tensorboard', '--logdir=~/Dropbox/Programming/Python/Attitude_1/train'],
                        stdout=PIPE, stderr=PIPE)
    time.sleep(5)
    webbrowser.open('http://0.0.0.0:6006')
    while input('Press <q> to quit') != 'q':
        continue
    tensorboard.terminate()

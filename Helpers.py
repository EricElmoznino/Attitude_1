import tensorflow as tf
import numpy as np
import time
from PIL import Image
import math
import webbrowser
from subprocess import Popen, PIPE

def open_tensorboard():
    tensorboard = Popen(['tensorboard', '--logdir=~/Dropbox/Programming/Python/Attitude_1/train'],
                        stdout=PIPE, stderr=PIPE)
    time.sleep(5)
    webbrowser.open('http://0.0.0.0:6006')
    while input('Press <q> to quit') != 'q':
        continue
    tensorboard.terminate()

def log_step(step, total_steps, start_time, angle_error):
    progress = int(step/float(total_steps) * 100)

    seconds = time.time() - start_time
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    print(str(progress) + '%\t|\t',
          int(h), 'hours,', int(m), 'minutes,', int(s), 'seconds\t|\t',
          'Step:', step, '/', total_steps, '\t|\t',
          'Average Angle Error (Degrees):', angle_error*180/math.pi)

def log_epoch(epoch, total_epochs, angle_error):
    print('\nEpoch', epoch, 'completed out of', total_epochs,
          '\t|\tAverage Angle Error (Degrees):', angle_error*180/math.pi)

def log_generic(angle_error, set_name):
    print('Average Angle Error (Degrees) on', set_name, 'set:', angle_error*180/math.pi, '\n')

def images_to_batch(image_data):
    batch_input = []
    batch_labels = []
    for file_l, file_r, attitude in image_data:
        input_left = (np.asarray(Image.open(file_l)) / 255).flatten()
        input_right = (np.asarray(Image.open(file_r)) / 255).flatten()
        batch_input.append(np.concatenate((input_left, input_right)))
        batch_labels.append(attitude)
    return batch_input, batch_labels

def weight_variables(shape):
    initial = tf.truncated_normal_initializer(stddev=0.1)
    return tf.get_variable('weights', shape=shape,
                           initializer=initial)

def bias_variables(shape):
    initial = tf.constant_initializer(0.1)
    return tf.get_variable('biases', shape=shape,
                           initializer=initial)

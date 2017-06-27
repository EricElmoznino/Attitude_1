import tensorflow as tf
import numpy as np
import time
import os
import webbrowser
from subprocess import Popen, PIPE


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
    return list(zip(files_l, files_r, attitudes))


def image_input_queue(data, img_shape, label_shape, batch_size=50):
    file_paths_left, file_paths_right, labels = list(zip(*data))
    file_paths_left_t = tf.convert_to_tensor(file_paths_left, dtype=tf.string)
    file_paths_right_t = tf.convert_to_tensor(file_paths_right, dtype=tf.string)
    labels_t = tf.convert_to_tensor(labels, dtype=tf.float32)

    input_queue = tf.train.slice_input_producer(
        [file_paths_left_t, file_paths_right_t, labels_t],
        shuffle=False)
    file_content_left = tf.read_file(input_queue[0])
    file_content_right = tf.read_file(input_queue[1])
    left_img = tf.image.decode_jpeg(file_content_left, channels=img_shape[2])
    right_img = tf.image.decode_jpeg(file_content_right, channels=img_shape[2])
    left_img = tf.cast(left_img, dtype=tf.float32)
    left_img = left_img / 255
    right_img = tf.cast(right_img, dtype=tf.float32)
    right_img = right_img / 255
    label = input_queue[2]

    left_img.set_shape(img_shape)
    right_img.set_shape(img_shape)
    label.set_shape(label_shape)

    min_queue_examples = 256
    input_queue = tf.train.shuffle_batch(
        [left_img, right_img, label],
        batch_size=batch_size,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples
    )

    return input_queue


def process_data(train_path, test_path, validation_path, batch_size, img_shape, label_shape):
    train_data = data_at_path(train_path)
    test_data = data_at_path(test_path)
    validation_data = data_at_path(validation_path)

    train_queue = image_input_queue(train_data,
                                    img_shape=img_shape, label_shape=label_shape,
                                    batch_size=batch_size)
    test_queue = image_input_queue(test_data,
                                   img_shape=img_shape, label_shape=label_shape)
    validation_queue = image_input_queue(validation_data,
                                         img_shape=img_shape, label_shape=label_shape)

    return train_queue, test_queue, validation_queue


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


def weight_variables(shape):
    initial = tf.truncated_normal_initializer(stddev=0.1)
    return tf.get_variable('weights', shape=shape,
                           initializer=initial)


def bias_variables(shape):
    initial = tf.constant_initializer(0.1)
    return tf.get_variable('biases', shape=shape,
                           initializer=initial)


def convolve(model, window, n_inputs, n_outputs, stride=None):
    with tf.variable_scope('convolution'):
        if stride is None:
            stride = [1, 1]
        weights = weight_variables(window + [n_inputs] + [n_outputs])
        biases = bias_variables([n_outputs])
        stride = [1] + stride + [1]
        return tf.nn.conv2d(model, weights, stride, padding='SAME') + biases


def max_pool(model, pool_size):
    stride = [1] + pool_size + [1]
    return tf.nn.max_pool(model, ksize=stride, strides=stride, padding='SAME')


def open_tensorboard():
    tensorboard = Popen(['tensorboard', '--logdir=~/Dropbox/Programming/Python/Attitude_1/train'],
                        stdout=PIPE, stderr=PIPE)
    time.sleep(5)
    webbrowser.open('http://0.0.0.0:6006')
    while input('Press <q> to quit') != 'q':
        continue
    tensorboard.terminate()

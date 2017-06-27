import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import Helpers as hp
import numpy as np
import shutil
import os
import time

epochs = 10
batch_size = 50
img_shape = [100, 100, 3]
label_shape = [3]

left_img_placeholder = tf.placeholder(tf.float32, shape=[None] + img_shape, name='left_eye')
right_img_placeholder = tf.placeholder(tf.float32, shape=[None] + img_shape, name='right_eye')
labels_placeholder = tf.placeholder(tf.float32, shape=[None] + label_shape, name='labels')
with tf.variable_scope('hyperparameters'):
    keep_prob_placeholder = tf.placeholder(tf.float32, name='dropout_keep_probability')

train_path = './train'
train_data_path = '/Users/Eric/Desktop/train_data'
test_data_path = '/Users/Eric/Desktop/test_data'
validation_data_path = '/Users/Eric/Desktop/validation_data'


def loss_for_queue(sess, loss, queue, n_batches):
    error = 0
    for batch in range(n_batches):
        left_img, right_img, labels = sess.run(queue)
        error += sess.run(loss,
                          feed_dict={left_img_placeholder: left_img,
                                     right_img_placeholder: right_img,
                                     labels_placeholder: labels,
                                     keep_prob_placeholder: 1.0})
    return error / n_batches


def embeddings_for_queue(sess, model, queue, n_batches, train_path):
    predictions = np.ndarray([0, 3])
    prediction_labels = np.ndarray([0, 3])
    for batch in range(n_batches):
        left_img, right_img, labels = sess.run(queue)
        prediction = sess.run(model,
                              feed_dict={left_img_placeholder: left_img,
                                         right_img_placeholder: right_img,
                                         labels_placeholder: labels,
                                         keep_prob_placeholder: 1.0})
        predictions = np.concatenate((predictions, prediction))
        prediction_labels = np.concatenate((prediction_labels, labels))

    with tf.variable_scope('embedding'):
        embedding_var = tf.get_variable('embedding_var', shape=[predictions.shape[0], predictions.shape[1]],
                                        initializer=tf.constant_initializer(predictions))
    sess.run(embedding_var.initializer)

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    metadata_file_path = os.path.join(train_path, 'metadata.tsv')
    embedding.metadata_path = metadata_file_path

    # data labels
    with open(metadata_file_path, 'w') as f:
        for label in prediction_labels:
            f.write(str(label) + '\n')

    writer = tf.summary.FileWriter(train_path)
    projector.visualize_embeddings(writer, config)
    embed_saver = tf.train.Saver([embedding_var])
    embed_saver.save(sess, os.path.join(train_path, 'embeddding.ckpt'))


def build_model_conv():
    filter_sizes = [10]
    features_sizes = [15]
    hidden_sizes = [1000]

    with tf.variable_scope('model'):

        with tf.variable_scope('convolution'):
            layer = 1
            left_units = left_img_placeholder
            right_units = right_img_placeholder
            for filter_size, feature_size in zip(filter_sizes, features_sizes):
                with tf.variable_scope('convolution_layer_' + str(layer)) as scope:
                    left_units = hp.convolve(left_units, [filter_size, filter_size],
                                             left_units.shape[-1], feature_size)
                    left_units = tf.nn.relu(left_units)
                    scope.reuse_variables()
                    right_units = hp.convolve(right_units, [filter_size, filter_size],
                                              right_units.shape[-1], feature_size)
                    right_units = tf.nn.relu(right_units)
                layer += 1

        with tf.variable_scope('fully_connected'):
            layer = 1
            num_units = left_units.shape[1] * left_units.shape[2] * left_units.shape[3]
            left_units = tf.reshape(left_units, [-1, int(num_units)])
            right_units = tf.reshape(right_units, [-1, int(num_units)])
            hidden_units = tf.concat([left_units, right_units], axis=1)
            for hidden_size in hidden_sizes:
                with tf.variable_scope('hidden_layer_' + str(layer)):
                    weights = hp.weight_variables([hidden_units.shape[1], hidden_size])
                    biases = hp.bias_variables([hidden_size])
                    hidden_units = tf.matmul(hidden_units, weights) + biases
                    hidden_units = tf.nn.relu(hidden_units)
                layer += 1

        with tf.variable_scope('output'):
            weights = hp.weight_variables([hidden_units.shape[1], 3])
            model = tf.matmul(hidden_units, weights)
            model = tf.nn.dropout(model, keep_prob=keep_prob_placeholder)

    return model, tf.train.Saver()


def build_model_normal():
    split_sizes = [1000, 500]
    merged_sizes = [1000, 200]

    with tf.variable_scope('model'):

        with tf.variable_scope('split'):
            layer = 1
            num_units = img_shape[0] * img_shape[1] * img_shape[2]
            left_units = tf.reshape(left_img_placeholder, [-1, num_units])
            right_units = tf.reshape(right_img_placeholder, [-1, num_units])
            for split_size in split_sizes:
                with tf.variable_scope('layer_' + str(layer)):
                    weights = hp.weight_variables([left_units.shape[1], split_size])
                    biases = hp.bias_variables([split_size])
                    left_units = tf.matmul(left_units, weights) + biases
                    left_units = tf.nn.relu(left_units)
                    right_units = tf.matmul(right_units, weights) + biases
                    right_units = tf.nn.relu(right_units)
                layer += 1

        with tf.variable_scope('merged'):
            layer = 1
            merged_units = tf.concat([left_units, right_units], axis=1)
            for merged_size in merged_sizes:
                with tf.variable_scope('layer_' + str(layer)):
                    weights = hp.weight_variables([merged_units.shape[1], merged_size])
                    biases = hp.bias_variables([merged_size])
                    merged_units = tf.matmul(merged_units, weights) + biases
                    merged_units = tf.nn.relu(merged_units)
                layer += 1

        with tf.variable_scope('output'):
            weights = hp.weight_variables([merged_units.shape[1], 3])
            model = tf.matmul(merged_units, weights)
            model = tf.nn.dropout(model, keep_prob=keep_prob_placeholder)

    return model, tf.train.Saver()

def build_model_simple():
    with tf.variable_scope('model'):
        left_units = tf.reshape(left_img_placeholder, [-1, 30000])
        right_units = tf.reshape(right_img_placeholder, [-1, 30000])
        layer = tf.concat([left_units, right_units], axis=1)
        weights = hp.weight_variables([60000, 3])
        model = tf.matmul(layer, weights)
        model = tf.nn.dropout(model, keep_prob=keep_prob_placeholder)
    return model, tf.train.Saver()


def train(model, saver):
    epochs = 1
    batch_size = 50
    keep_prob = 0.5
    train_samples = 10000
    test_samples = 1000
    validation_samples = 200

    train_set, test_set, validation_set = hp.process_data(train_data_path, test_data_path, validation_data_path,
                                                          batch_size, img_shape, label_shape)

    n_batches = int(train_samples / batch_size)
    n_steps = n_batches * epochs

    with tf.variable_scope('training'):
        sqr_dif = tf.reduce_sum(tf.square(model - labels_placeholder), 1)
        mse = tf.reduce_mean(sqr_dif, name='mean_squared_error')
        angle_error = tf.reduce_mean(tf.sqrt(sqr_dif), name='mean_angle_error')
        tf.summary.scalar('angle_error', angle_error)
        optimizer = tf.train.AdamOptimizer().minimize(mse)

    summaries = tf.summary.merge_all()
    if os.path.exists(train_path):
        shutil.rmtree(train_path)
    os.mkdir(train_path)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter(train_path, sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        time.sleep(20)  # let the queues fill up a bit initially

        print('Starting training\n')
        start_time = time.time()
        step = 0
        for epoch in range(1, epochs + 1):
            epoch_angle_error = 0
            for batch in range(n_batches):
                left_img, right_img, labels = sess.run(train_set)

                if step % max(int(n_steps / 1000), 1) == 0:
                    _, a, s = sess.run([optimizer, angle_error, summaries],
                                       feed_dict={left_img_placeholder: left_img,
                                                  right_img_placeholder: right_img,
                                                  labels_placeholder: labels,
                                                  keep_prob_placeholder: keep_prob})
                    train_writer.add_summary(s, step)
                    hp.log_step(step, n_steps, start_time, a)
                else:
                    _, a = sess.run([optimizer, angle_error],
                                    feed_dict={left_img_placeholder: left_img,
                                               right_img_placeholder: right_img,
                                               labels_placeholder: labels,
                                               keep_prob_placeholder: keep_prob})

                epoch_angle_error += a
                step += 1

            hp.log_epoch(epoch, epochs, epoch_angle_error / n_batches)
            val_angle_error = loss_for_queue(sess, angle_error, validation_set, int(validation_samples / 50))
            hp.log_generic(val_angle_error, 'validation')

        test_angle_error = loss_for_queue(sess, angle_error, test_set, int(test_samples / 50))
        hp.log_generic(test_angle_error, 'test')
        saver.save(sess, os.path.join(train_path, 'model.ckpt'))

        embeddings_for_queue(sess, model, test_set, int(test_samples / 50), train_path)

        coord.request_stop()
        coord.join(threads)


###############################################################################
# Main script

att_model, att_saver = build_model_conv()
train(att_model, att_saver)

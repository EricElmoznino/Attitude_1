import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import Helpers as hp
import numpy as np
import shutil
import os
import time

epochs = 10
batch_size = 50
input_size = 60000
split_sizes = [1000, 500]
merged_sizes = [1000, 200]


input_data = tf.placeholder(tf.float32, shape=[None, input_size], name='input')
output_labels = tf.placeholder(tf.float32, shape=[None, 3], name='labels')
with tf.variable_scope('hyperparameters'):
    keep_prob = tf.placeholder(tf.float32, name='dropout_keep_probability')

train_path = os.path.join('./', 'train')


def run_scalar_in_batches(sess, model, inputs, labels):
    nBatches = int(len(inputs) / batch_size)
    batch_inputs = [inputs[i*batch_size:(i+1)*batch_size] for i in range(nBatches)]
    batch_labels = [labels[i*batch_size:(i+1)*batch_size] for i in range(nBatches)]
    res = 0
    for inp, lbl in zip(batch_inputs, batch_labels):
        res += sess.run(model,
                        feed_dict={input_data: inp,
                                   output_labels: lbl,
                                   keep_prob: 1.0})
    return res/nBatches

def create_embeddings(sess, model, inputs, labels, train_path):
    predictions = np.ndarray([0, model.shape[1]])
    for inp in inputs:
        prediction = sess.run(model,
                              feed_dict={input_data: [inp],
                                         keep_prob: 1.0})
        predictions = np.concatenate((predictions, prediction))

    with tf.variable_scope('embedding'):
        embedding_var = tf.get_variable('embedding_var', shape=[predictions.shape[0], predictions.shape[1]],
                                        initializer=tf.constant_initializer(predictions))
    sess.run(embedding_var.initializer)

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    metadata_file_path = os.path.join('./', 'metadata.tsv')
    embedding.metadata_path = metadata_file_path

    # data labels
    if os.path.exists(metadata_file_path):
        os.remove(metadata_file_path)
    with open(metadata_file_path, 'w') as f:
        for label in labels:
            f.write(str(label) + '\n')

    writer = tf.summary.FileWriter(train_path)
    emb_saver = tf.train.Saver([embedding_var])
    projector.visualize_embeddings(writer, config)
    emb_saver.save(sess, os.path.join(train_path, 'embedding_model.ckpt'))

def build_model():
    with tf.variable_scope('model'):
        input_data_lr = tf.split(input_data, 2, axis=1)
        with tf.variable_scope('layer_1'):
            weights = hp.weight_variables([input_size/2, split_sizes[0]])
            biases = hp.bias_variables([split_sizes[0]])
            hl1 = tf.matmul(input_data_lr[0], weights) + biases
            hl1 = tf.nn.relu(hl1)
            hr1 = tf.matmul(input_data_lr[1], weights) + biases
            hr1 = tf.nn.relu(hr1)
        with tf.variable_scope('layer_2'):
            weights = hp.weight_variables([split_sizes[0], split_sizes[1]])
            biases = hp.bias_variables([split_sizes[1]])
            hl2 = tf.matmul(hl1, weights) + biases
            hl2 = tf.nn.relu(hl2)
            hr2 = tf.matmul(hr1, weights) + biases
            hr2 = tf.nn.relu(hr2)
        with tf.variable_scope('layer_3'):
            weights = hp.weight_variables([split_sizes[1], merged_sizes[0]])
            biases = hp.bias_variables([merged_sizes[0]])
            h3 = tf.matmul(hl2, weights) + tf.matmul(hr2, weights) + biases
            h3 = tf.nn.relu(h3)
        with tf.variable_scope('layer_4'):
            weights = hp.weight_variables([merged_sizes[0], merged_sizes[1]])
            biases = hp.bias_variables([merged_sizes[1]])
            h4 = tf.matmul(h3, weights) + biases
            h4 = tf.nn.relu(h4)
        with tf.variable_scope('layer_5'):
            weights = hp.weight_variables([merged_sizes[1], 3])
            biases = hp.bias_variables([3])
            output = tf.matmul(h4, weights) + biases
        output = tf.nn.dropout(output, keep_prob=keep_prob)
    return output

def train(model, train_data, validation_data, test_data):
    nBatches = int(len(train_data) / batch_size)
    nSteps = nBatches*epochs
    batches = [train_data[i*batch_size:(i+1)*batch_size] for i in range(nBatches)]

    with tf.variable_scope('training'):
        sqr_dif = tf.reduce_sum(tf.square(model - output_labels), 1)
        mse = tf.reduce_mean(sqr_dif, name='mean_squared_error')
        angle_error = tf.reduce_mean(tf.sqrt(sqr_dif), name='mean_angle_error')
        tf.summary.scalar('angle_error', angle_error)
        optimizer = tf.train.AdamOptimizer().minimize(mse)

    summaries = tf.summary.merge_all()
    if os.path.exists(train_path):
        shutil.rmtree(train_path)
    os.mkdir(train_path)

    saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter(train_path, sess.graph)

        start_time = time.time()
        step = 0
        for epoch in range(1, epochs+1):
            epoch_angle_error = 0

            for batch in batches:
                batch_input, batch_labels = hp.images_to_batch(batch)

                if step % max(int(nSteps/1000), 1) == 0:
                    _, a, s = sess.run([optimizer, angle_error, summaries],
                                       feed_dict={input_data: batch_input,
                                                  output_labels: batch_labels,
                                                  keep_prob: 0.5})
                    train_writer.add_summary(s, step)
                    saver.save(sess, os.path.join(train_path, 'model.ckpt'), global_step=step)
                    log_step(step, nSteps, start_time, a)
                else:
                    _, a = sess.run([optimizer, angle_error],
                                    feed_dict={input_data: batch_input,
                                               output_labels: batch_labels,
                                               keep_prob: 0.5})

                epoch_angle_error += a
                step += 1

            hp.log_epoch(epoch, epochs, epoch_angle_error/nBatches)
            val_input, val_labels = hp.images_to_batch(validation_data)
            val_angle_error = run_scalar_in_batches(sess, angle_error, val_input, val_labels)
            hp.log_generic(val_angle_error, 'validation')

        test_input, test_labels = hp.images_to_batch(test_data)
        test_angle_error = run_scalar_in_batches(sess, angle_error, test_input, test_labels)
        hp.log_generic(test_angle_error, 'test')
        saver.save(sess, os.path.join(train_path, 'model.ckpt'))

        create_embeddings(sess, model, test_input, test_labels, train_path)

def predict(model, data):
    model_input, _ = hp.images_to_batch(data)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        try:
            saver.restore(sess, os.path.join(train_path, 'model.ckpt'))
        except Exception as e:
            print(str(e))

        attitude = sess.run([model],
                            feed_dict={input_data: model_input,
                                       keep_prob: 1.0})

    return attitude

###############################################################################
# Main script

# Create the train, validation, and test data lists
train_data_path = '/home/eric/Desktop/train_data'
valid_data_path = '/home/eric/Desktop/validation_data'
test_data_path = '/home/eric/Desktop/test_data'
pred_data_path = '/home/eric/Desktop/prediction_data'

def process_image_files(path):
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

train_data = process_image_files(train_data_path)
valid_data = process_image_files(valid_data_path)
test_data = process_image_files(test_data_path)
pred_data = process_image_files(pred_data_path)

model = build_model()
train(model, train_data, valid_data, test_data)
res = predict(model, pred_data)
print(res)


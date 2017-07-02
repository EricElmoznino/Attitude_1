import tensorflow as tf
import tensorflow.contrib.data as data
from tensorflow.contrib.tensorboard.plugins import projector
import Helpers as hp
import numpy as np
import shutil
import os
import time

class Model:

    def __init__(self, configuration, image_width, image_height):
        self.conf = configuration

        self.image_shape = [image_width, image_height, 3]
        self.label_shape = [3]

        with tf.variable_scope('hyperparameters'):
            self.keep_prob_placeholder = tf.placeholder(tf.float32, name='dropout_keep_probability')

        self.dataset_placeholders, self.datasets, self.iterator = self.create_input_pipeline()
        self.left_images, self.right_images, self.labels = self.iterator.get_next()
        self.model = self.build_model()
        self.saver = tf.train.Saver()

    def create_input_pipeline(self):
        with tf.variable_scope('input_pipeline'):
            left_img_files = tf.placeholder(tf.string, [None])
            right_img_files = tf.placeholder(tf.string, [None])
            labels = tf.placeholder(tf.float32, [None] + self.label_shape)
            placeholders = {'left': left_img_files, 'right': right_img_files, 'labels': labels}

            def process_images(left_img_file, right_img_file, label):
                left_img_content = tf.read_file(left_img_file)
                right_img_content = tf.read_file(right_img_file)
                left_img = tf.image.decode_jpeg(left_img_content, channels=self.image_shape[-1])
                right_img = tf.image.decode_jpeg(right_img_content, channels=self.image_shape[-1])
                left_img = tf.divide(tf.cast(left_img, tf.float32), 255)
                right_img = tf.divide(tf.cast(right_img, tf.float32), 255)
                left_img.set_shape(self.image_shape)
                right_img.set_shape(self.image_shape)
                return left_img, right_img, label

            dataset = data.Dataset.from_tensor_slices((left_img_files, right_img_files, labels))
            dataset = dataset.map(process_images)
            dataset = dataset.repeat()
            train_set = dataset.batch(self.conf.batch_size)
            predict_set = dataset.batch(1)
            datasets = {'train': train_set, 'predict': predict_set}

            iterator = data.Iterator.from_dataset(train_set)

        return placeholders, datasets, iterator

    def build_model(self):
        with tf.variable_scope('model'):
            with tf.variable_scope('convolution_layer_1') as scope:
                left_units = hp.convolve(self.left_images, [5, 5], 3, 20, stride=[2, 2])
                left_units = tf.nn.relu(left_units)
                left_units = hp.max_pool(left_units, [2, 2])
                scope.reuse_variables()
                right_units = hp.convolve(self.right_images, [5, 5], 3, 20, stride=[2, 2])
                right_units = tf.nn.relu(right_units)
                right_units = hp.max_pool(right_units, [2, 2])
            with tf.variable_scope('fully_connected_layer_1'):
                left_units = tf.reshape(left_units, [-1, 24*24*20])
                right_units = tf.reshape(right_units, [-1, 24*24*20])
                units = tf.concat([left_units, right_units], axis=1)
                weights = hp.weight_variables([2*24*24*20, 5000])
                biases = hp.bias_variables([5000])
                units = tf.add(tf.matmul(units, weights), biases)
                units = tf.nn.relu(units)
            with tf.variable_scope('output_layer'):
                weights = hp.weight_variables([5000, 3], mean=0.0)
                model = tf.matmul(units, weights)
                model = tf.nn.dropout(model, keep_prob=self.keep_prob_placeholder)
        return model

    def train(self, train_path, validation_path = None, test_path = None):
        with tf.variable_scope('training'):
            sqr_dif = tf.reduce_sum(tf.square(self.model - self.labels), 1)
            mse = tf.reduce_mean(sqr_dif, name='mean_squared_error')
            angle_error = tf.reduce_mean(tf.sqrt(sqr_dif), name='mean_angle_error')
            tf.summary.scalar('angle_error', angle_error)
            optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(mse)

        summaries = tf.summary.merge_all()
        if os.path.exists(self.conf.train_log_path):
            shutil.rmtree(self.conf.train_log_path)
        os.mkdir(self.conf.train_log_path)

        print('Starting training\n')
        with tf.Session(config=tf.ConfigProto(device_count={'GPU': 1})) as sess:
            sess.run(tf.global_variables_initializer())
            train_writer = tf.summary.FileWriter(self.conf.train_log_path, sess.graph)

            start_time = time.time()
            step = 0
            for epoch in range(1, self.conf.epochs + 1):
                epoch_angle_error = 0
                n_samples = self.initialize_iterator_with_set(sess, train_path, 'train')
                n_batches = int(n_samples / self.conf.batch_size)
                n_steps = n_batches * self.conf.epochs

                for batch in range(n_batches):
                    if step % max(int(n_steps / 1000), 1) == 0:
                        _, a, s = sess.run([optimizer, angle_error, summaries],
                                           feed_dict={self.keep_prob_placeholder: self.conf.keep_prob})
                        train_writer.add_summary(s, step)
                        hp.log_step(step, n_steps, start_time, a)
                    else:
                        _, a = sess.run([optimizer, angle_error],
                                        feed_dict={self.keep_prob_placeholder: self.conf.keep_prob})

                    epoch_angle_error += a
                    step += 1

                hp.log_epoch(epoch, self.conf.epochs, epoch_angle_error / n_batches)
                if validation_path is not None:
                    self.error_for_set(sess, angle_error, validation_path, 'validation')

            self.saver.save(sess, os.path.join(self.conf.train_log_path, 'model.ckpt'))
            if test_path is not None:
                self.error_for_set(sess, angle_error, test_path, 'test')
                self.embeddings_for_set(sess, test_path)

    def predict(self, prediction_path):
        with tf.Session() as sess:
            try:
                self.saver.restore(sess, os.path.join(self.conf.train_log_path, 'model.ckpt'))
            except Exception as e:
                print(str(e))

            n_samples = self.initialize_iterator_with_set(sess, prediction_path, 'predict')
            predictions = np.ndarray([0, 3])
            for _ in range(n_samples):
                prediction = sess.run(self.model, feed_dict={self.keep_prob_placeholder: 1.0})
                predictions = np.concatenate([predictions, prediction])

        return predictions

    def error_for_set(self, sess, error, path, name):
        n_samples = self.initialize_iterator_with_set(sess, path, 'predict')
        average_error = 0
        for _ in range(n_samples):
            average_error += sess.run(error, feed_dict={self.keep_prob_placeholder: 1.0}) / n_samples
        hp.log_generic(average_error, name)
        return average_error

    def embeddings_for_set(self, sess, path):
        n_samples = self.initialize_iterator_with_set(sess, path, 'predict')
        predictions = np.ndarray([0, 3])
        prediction_labels = np.ndarray([0, 3])
        for _ in range(n_samples):
            prediction, label = sess.run([self.model, self.labels], feed_dict={self.keep_prob_placeholder: 1.0})
            predictions = np.concatenate([predictions, prediction])
            prediction_labels = np.concatenate([prediction_labels, label])

        with tf.variable_scope('embedding'):
            embedding_var = tf.get_variable('embedding_var', shape=[predictions.shape[0], predictions.shape[1]],
                                            initializer=tf.constant_initializer(predictions))
        sess.run(embedding_var.initializer)

        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name
        metadata_file_path = os.path.join(self.conf.train_log_path, 'metadata.tsv')
        embedding.metadata_path = metadata_file_path

        # data labels
        with open(metadata_file_path, 'w') as f:
            for label in prediction_labels:
                f.write(str(label) + '\n')

        writer = tf.summary.FileWriter(self.conf.train_log_path)
        projector.visualize_embeddings(writer, config)
        embed_saver = tf.train.Saver([embedding_var])
        embed_saver.save(sess, os.path.join(self.conf.train_log_path, 'embeddding.ckpt'))

    def initialize_iterator_with_set(self, sess, path, type):
        files_left, files_right, labels = hp.data_at_path(path)
        init = self.iterator.make_initializer(self.datasets[type])
        sess.run(init,
                 feed_dict={self.dataset_placeholders['left']: files_left,
                            self.dataset_placeholders['right']: files_right,
                            self.dataset_placeholders['labels']: labels})
        return len(files_left)



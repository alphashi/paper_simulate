import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

import cifar10
import cifar10_input

from datetime import datetime
import time

import nets.vgg as vgg


def nin(inputs):

    with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        padding="SAME",
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.01)):

        const_value = 0.0
        # const_init = weights_initializer=tf.constant_initializer(const_value)
        const_init = weights_initializer=tf.truncated_normal_initializer(mean=const_value, stddev=0.05)
        # layer 1
        net0 = slim.conv2d(inputs, num_outputs=192, kernel_size=[5, 5], stride = 1, scope="conv1")
        net1 = slim.conv2d(net0, num_outputs=169, kernel_size=[1, 1], weights_initializer=const_init, scope="cccp1")
        net2 = slim.conv2d(net1, num_outputs=96, kernel_size=[1, 1], weights_initializer=const_init, scope="cccp2")
        net = slim.max_pool2d(net2, kernel_size=[3, 3], stride=2, scope="pool1")

        # layer 2
        net = slim.conv2d(net, num_outputs=192, kernel_size=[5, 5], stride = 1, scope="conv2")
        net = slim.conv2d(net, num_outputs=192, kernel_size=[1, 1], weights_initializer=const_init, scope="cccp3")
        net = slim.conv2d(net, num_outputs=192, kernel_size=[1, 1], weights_initializer=const_init, scope="cccp4")
        net = slim.max_pool2d(net, kernel_size=[3, 3], stride=1, scope="pool2")

        # layer 3
        net = slim.conv2d(net, num_outputs=192, kernel_size=[5, 5], stride = 1, scope="conv3")
        net = slim.conv2d(net, num_outputs=192, kernel_size=[1, 1], weights_initializer=const_init, scope="cccp5")
        net = slim.conv2d(net, num_outputs=10, kernel_size=[1, 1], weights_initializer=const_init, scope="cccp6")
        net = slim.avg_pool2d(net, kernel_size=[13, 13], stride=1, scope="pool3")
        # net = slim.avg_pool2d(net, kernel_size=[9, 9], stride=1, scope="pool3")

        slim.summaries.add_histogram_summaries(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

        net = slim.flatten(net)

    # return tf.reshape(net, [None, 10])

    return net


def mlp(inputs):

    with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        padding="SAME",
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.01)):

        net = slim.conv2d(inputs, num_outputs=192, kernel_size=[5, 5], stride = 1, scope="conv1")
        net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope="pool1")

        # layer 2
        net = slim.conv2d(net, num_outputs=192, kernel_size=[5, 5], stride = 1, scope="conv2")
        net = slim.conv2d(net, num_outputs=192, kernel_size=[1, 1], weights_initializer=tf.constant_initializer(1.0), scope="cccp3")
        # net = slim.conv2d(net, num_outputs=192, kernel_size=[1, 1], weights_initializer=tf.constant_initializer(1.0), scope="cccp4")
        net = slim.max_pool2d(net, kernel_size=[3, 3], stride=1, scope="pool2")

        input = slim.flatten(net)
        net0 = slim.fully_connected(input, 500, scope='fc1')
        net1 = slim.fully_connected(net0, 200, scope='fc2')
        net2 = slim.fully_connected(net1, 10, activation_fn=None, scope='fc3')

        # net = slim.flatten(net2)

    return net2, net0, net1


def accuracy(images, labels, batch_size, top_k=1):

    top_k_op = tf.nn.in_top_k(predictions, labels, top_k)
    accur = tf.reduce_sum(tf.cast(top_k_op, tf.int32), reduction_indices=0) / batch_size

    return accur


def evaluate(top_k_op, num_examples, batch_size):
    with tf.Session() as sess:

        coord = tf.train.Coordinator()

        num_iter = int(np.ceil(num_examples / batch_size))
        true_count = 0  # Counts the number of correct predictions.
        total_sample_count = num_iter * batch_size
        step = 0

        while step < num_iter and not coord.should_stop():
            predictions = sess.run([top_k_op])
            true_count += np.sum(predictions)
            step += 1

        precision = true_count / total_sample_count
        print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))


if __name__ == "__main__":

    log_frequency = 1
    cifar10.BATCH_SIZE = 16
    batch_size = cifar10.BATCH_SIZE

    num_examples = 10000

    print ("batch size:", batch_size)

    # load cifar10 data
    cifar10.maybe_download_and_extract()

      # predictions = nin(images)

      # loss = slim.losses.softmax_cross_entropy(predictions, labels)

    train_log_dir = "train_log_" + datetime.now().strftime('%Y%m%d_%H%M%S')
    test_data_name = ["test_batch_bin"]

    if not tf.gfile.Exists(train_log_dir):
        tf.gfile.MakeDirs(train_log_dir)

    with tf.Graph().as_default():
          # Set up the data loading:
          # images, labels = ...
        with tf.device('/cpu:0'):
            images, labels = cifar10.distorted_inputs()

            test_images, test_labels = cifar10.inputs(True)
            # test_images, test_labels = cifar10.inputs

            # filename_queue = tf.train.string_input_producer(test_data_name)
            # read_input = cifar10_input.read_cifar10(filename_queue)
            # reshaped_image = tf.cast(read_input.uint8image, tf.float32)
            #
            # read_input.label.set_shape([1])
            # test_labels = read_input.label
        # batch_size = images.shape()[0]

          # Define the model:
        # predictions, net0, net1 = mlp(images)
        predictions = nin(images)
        accur = accuracy(predictions, labels, batch_size)

        # accuracy = tf.contrib.metrics.streaming_sparse_precision_at_k(predictions, tf.cast(labels, tf.int64), 1)
        # top_k_op = tf.nn.in_top_k(predictions, labels, 1)
        # accuracy = tf.reduce_sum(tf.cast(top_k_op, tf.int32), reduction_indices=0) / batch_size
        summ_accu = tf.summary.scalar('train_accuracy', accur)


        # infer = nin(test_images)
        # infer_accur = accuracy(infer, test_labels, batch_size)


        total_loss = tf.losses.sparse_softmax_cross_entropy(
                        labels, predictions, scope='cross_entropy_per_example', weights=1.0)

        # total_loss = total_loss + 0.00 * tf.losses.get_regularization_loss()
        # total_loss = tf.losses.get_total_loss()
        summ_loss = tf.summary.scalar('losses/total_loss', total_loss)
            # tf.summary.scalar("loss", loss)


        starter_learning_rate = 0.01
        # global_step = tf.train.get_global_step()
        # global_step = tf.Variable(0, trainable=False)
        global_step = tf.contrib.framework.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                     2, 0.80, staircase=True)

        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                                 momentum=0.9)  # .minimize(total_loss, global_step=global_step)

          # create_train_op that ensures that when we evaluate it to get the loss,
          # the update_ops are done and the gradient updates are computed.
          # train_tensor = slim.learning.create_train_op(total_loss, optimizer)

        train_tensor = tf.contrib.training.create_train_op(total_loss, optimizer, global_step=global_step,
                                                             summarize_gradients=True)
          # Actually runs training.



        class _LoggerHook(tf.train.SessionRunHook):
            """Logs loss and runtime."""

            def begin(self):
                self._step = -1
                self._start_time = time.time()

            def before_run(self, run_context):
                self._step += 1
                return tf.train.SessionRunArgs([total_loss, predictions, accur, labels, learning_rate])  # Asks for loss value.

            def after_run(self, run_context, run_values):
                if self._step % log_frequency == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time

                    loss_value = run_values.results[0]
                    pred_value = run_values.results[1]
                    accuracy_value = run_values.results[2]
                    label_value = run_values.results[3]
                    learning_rate_value = run_values.results[4]

                    # net0_value = run_values.results[1]
                    # net1_value = run_values.results[2]

                    # max_index = np.argmax(pred_value, axis=1)
                    # accu_value = np.sum(max_index == label_value) / batch_size
                    format_str = ("\n==========================================================\n"
                                  "%s: step %5d, loss = %.2f, accuracy = %.2f, learning_rate_value = %.4f, cost_time = %.2f")
                    print (format_str % (datetime.now(), self._step, loss_value, accuracy_value, learning_rate_value, duration))


                    evaluate(infer_accur, num_examples, batch_size)
                    # format_str = ("\n==========================================================\n"
                    #               "%s: step %5d, loss = %.2f, accuracy = %.2f == %.2f, cost_time = %.2f")
                    # print (format_str % (datetime.now(), self._step, loss_value, accu_value, accuracy_value, duration))
                    #
                    # examples_per_sec = log_frequency * batch_size / duration
                    # sec_per_batch = float(duration / log_frequency)
                    #
                    # format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                    #               'sec/batch)')
                    # print (format_str % (datetime.now(), self._step, loss_value,
                    #                      examples_per_sec, sec_per_batch))

          # Specify the optimization scheme:
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

        #learning_rate = 0.01


        # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9) #.minimize(total_loss, global_step=global_step)
        #
        #   # create_train_op that ensures that when we evaluate it to get the loss,
        #   # the update_ops are done and the gradient updates are computed.
        # # train_tensor = slim.learning.create_train_op(total_loss, optimizer)
        #
        # train_tensor = tf.contrib.training.create_train_op(total_loss, optimizer, global_step, summarize_gradients=True)
        #   # Actually runs training.

        tf.contrib.training.train(
                    train_tensor
                    , train_log_dir
                    , save_checkpoint_secs = 600
                    , save_summaries_steps = 100
                    # , hooks=[tf.train.StopAtStepHook(last_step=1000),
                    #          _LoggerHook()]
                    , hooks=[_LoggerHook()]
                    )

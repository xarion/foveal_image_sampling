import matplotlib;

matplotlib.use('agg')
import tensorflow as tf
from model import Model

import matplotlib.pyplot as plt
import os

import pickle

tf.logging.set_verbosity(tf.logging.INFO)

sigma = 1


class Train:
    def __init__(self, freeze_kernels, flag_d):
        self.output_dir = "data_" + str(freeze_kernels) + "_" + str(flag_d)

        self.pickle_data = {'freeze_kernels': freeze_kernels, 'flag_d': flag_d}

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.15)
        self.session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.checkpoint_dir = self.output_dir + "/checkpoints"

        with self.session.as_default():
            self.model = Model(sigma=sigma, freeze_kernels=freeze_kernels, flag_d=flag_d)
            self.train_writer = tf.summary.FileWriter(self.output_dir + "/train", self.session.graph)
            self.validation_writer = tf.summary.FileWriter(self.output_dir + "/validation", self.session.graph)
            self.merged_summaries = tf.summary.merge_all()
            self.saver = tf.train.Saver(max_to_keep=2)
            self.coord = tf.train.Coordinator()

    def initialize(self):
        self.session.run(tf.variables_initializer(tf.global_variables()))
        self.session.run(tf.variables_initializer(tf.local_variables()))

        latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
        if latest_checkpoint:
            tf.logging.info("loading from checkpoint file: " + latest_checkpoint)
            self.saver.restore(self.session, latest_checkpoint)
        else:
            tf.logging.info("checkpoint not found")

        tf.train.start_queue_runners(sess=self.session, coord=self.coord)

        if not os.path.exists(self.output_dir + "/kernels"):
            os.makedirs(self.output_dir + "/kernels")

    def run_training_batches(self):
        step = 0
        val_accuracy = 0.
        try:
            step, = self.session.run([self.model.global_step])
            while not self.coord.should_stop() and not (step >= 80000):
                if step % 100 == 0:
                    x_mus, y_mus, sigmas = self.session.run([self.model.x_mus, self.model.y_mus, self.model.sigmas])
                    self.plot_kernels(x_mus, y_mus, sigmas, step)
                # Train the model

                images, labels = self.model.get_numpy_data(mode=1)
                m, _, loss, step, labels, accuracy, = self.session.run([self.merged_summaries,
                                                                        self.model.training_op,
                                                                        self.model.loss,
                                                                        self.model.global_step,
                                                                        self.model.labels,
                                                                        self.model.accuracy],
                                                                       feed_dict={self.model.images: images,
                                                                                  self.model.labels: labels})
                self.train_writer.add_summary(m, step)

                # Do Validation every 20 steps
                if step % 20 == 0:
                    images, labels = self.model.get_numpy_data(mode=2)
                    m, val_accuracy, = self.session.run([self.merged_summaries, self.model.accuracy],
                                                        feed_dict={self.model.mode: 2,
                                                                   self.model.images: images,
                                                                   self.model.labels: labels})
                    self.validation_writer.add_summary(m, global_step=step)
                    tf.logging.info("===== step: %d, validation accuracy: %.2f =====" % (step, val_accuracy))

        except tf.errors.OutOfRangeError:
            tf.logging.info('Done training -- epoch limit reached')
        finally:
            self.save_checkpoint(step, val_accuracy)

    def save_checkpoint(self, step, accuracy):
        self.saver.save(self.session, self.checkpoint_dir + "model-%.2f" % accuracy, global_step=step)

    def finalize(self):
        self.validation_writer.flush()
        self.validation_writer.close()
        self.train_writer.flush()
        self.train_writer.close()
        self.coord.request_stop()
        self.coord.wait_for_stop()
        self.session.close()

    def train(self):
        self.initialize()
        self.run_training_batches()
        self.run_test_batch()
        self.pickle_for_sush()
        self.finalize()

    def run_test_batch(self):
        images, labels = self.model.get_numpy_data(mode=3)

        test_accuracy, = self.session.run([self.model.accuracy],
                                          feed_dict={self.model.mode: 3,
                                                     self.model.images: images,
                                                     self.model.labels: labels})
        self.pickle_data["test_accuracy"] = test_accuracy
        tf.logging.info("===== test accuracy %.2f =====" % test_accuracy)
        self.save_checkpoint(0, test_accuracy)

    def pickle_for_sush(self):
        x_mus, y_mus, sigmas, = self.session.run([self.model.x_mus, self.model.y_mus, self.model.sigmas])
        self.pickle_data["x_mus"] = x_mus
        self.pickle_data["y_mus"] = y_mus
        self.pickle_data["sigmas"] = sigmas
        pickle.dump(self.pickle_data, open(self.output_dir + ".pickle", "wb"), protocol=2)

    def plot_kernels(self, x_mus, y_mus, sigmas, step):
        x_mus = x_mus.flatten()
        y_mus = y_mus.flatten()
        sigmas = sigmas.flatten()
        fig, ax = plt.subplots()  # note we must use plt.subplots, not plt.subplot
        ax.set_facecolor("black")
        plt.xlim((0, 100))
        plt.ylim((0, 100))
        for x, y, sigma in zip(x_mus, y_mus, sigmas):
            ax.add_artist(plt.Circle((x, y), sigma, color='yellow'))
        fig.savefig(self.output_dir + '/kernels/%d.png' % step)


if __name__ == '__main__':
    t = Train(freeze_kernels=False, flag_d=2)
    t.train()

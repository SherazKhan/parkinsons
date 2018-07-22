import os, sys
import numpy as np
import tensorflow as tf
import datetime
from model import VggNetModel
sys.path.insert(0, '../utils')
from preprocessor import BatchPreprocessor


tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate for adam optimizer')
tf.app.flags.DEFINE_float('dropout_keep_prob', 0.5, 'Dropout keep probability')
tf.app.flags.DEFINE_integer('num_epochs', 1, 'Number of epochs for training')
tf.app.flags.DEFINE_integer('num_classes', 2, 'Number of classes')
tf.app.flags.DEFINE_integer('batch_size', 8, 'Batch size')
tf.app.flags.DEFINE_string('training_file', '../data/train.txt', 'Training dataset file')
tf.app.flags.DEFINE_string('val_file', '../data/test.txt', 'Validation dataset file')
tf.app.flags.DEFINE_string('tensorboard_root_dir', '../testing', 'Root directory to put the testing logs and weights')
tf.app.flags.DEFINE_integer('log_step', 10, 'Logging period in terms of iteration')
tf.app.flags.DEFINE_string('ckpt_path', '/home/acheketa/문서/workspace/parkinson/vgg/training/vggnet_20180220_134441/checkpoint/model_epoch38.ckpt', 'checkpoint path')

FLAGS = tf.app.flags.FLAGS


def main(_):
    # Create training directories
    now = datetime.datetime.now()
    train_dir_name = now.strftime('vggnet_%Y%m%d_%H%M%S')
    train_dir = os.path.join(FLAGS.tensorboard_root_dir, train_dir_name)
    checkpoint_dir = os.path.join(train_dir, 'checkpoint')
    tensorboard_dir = os.path.join(train_dir, 'tensorboard')
    tensorboard_train_dir = os.path.join(tensorboard_dir, 'train')
    tensorboard_val_dir = os.path.join(tensorboard_dir, 'val')

    if not os.path.isdir(FLAGS.tensorboard_root_dir): os.mkdir(FLAGS.tensorboard_root_dir)
    if not os.path.isdir(train_dir): os.mkdir(train_dir)
    if not os.path.isdir(checkpoint_dir): os.mkdir(checkpoint_dir)
    if not os.path.isdir(tensorboard_dir): os.mkdir(tensorboard_dir)
    if not os.path.isdir(tensorboard_train_dir): os.mkdir(tensorboard_train_dir)
    if not os.path.isdir(tensorboard_val_dir): os.mkdir(tensorboard_val_dir)

    # Placeholders
    img_size = 256
    x = tf.placeholder(tf.float32, [FLAGS.batch_size, img_size, img_size, 3])
    y = tf.placeholder(tf.float32, [None, FLAGS.num_classes])
    dropout_keep_prob = tf.placeholder(tf.float32)

    # Model
    #train_layers = FLAGS.train_layers.split(',')
    model = VggNetModel(num_classes=FLAGS.num_classes, dropout_keep_prob=dropout_keep_prob)
    loss = model.loss(x, y)
    #train_op = model.optimize(FLAGS.learning_rate, train_layers)
    train_op = model.optimize(FLAGS.learning_rate)

    # Training accuracy of the model
    correct_pred = tf.equal(tf.argmax(model.score, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Summaries
    tf.summary.scalar('train_loss', loss)
    tf.summary.scalar('train_accuracy', accuracy)
    merged_summary = tf.summary.merge_all()

    train_writer = tf.summary.FileWriter(tensorboard_train_dir)
    val_writer = tf.summary.FileWriter(tensorboard_val_dir)
    saver = tf.train.Saver()

    # Batch preprocessors
    val_preprocessor = BatchPreprocessor(dataset_file_path=FLAGS.val_file, num_classes=FLAGS.num_classes, output_size=[img_size, img_size])

    # Get the number of training/validation steps per epoch
    val_batches_per_epoch = np.floor(len(val_preprocessor.labels) / FLAGS.batch_size).astype(np.int16)

    test_accuracy = 0

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_writer.add_graph(sess.graph)

        # Directly restore (your model should be exactly the same with checkpoint)
        saver.restore(sess, FLAGS.ckpt_path)

        print("{} Start training...".format(datetime.datetime.now()))
        print("{} Open Tensorboard at --logdir {}".format(datetime.datetime.now(), tensorboard_dir))

        for epoch in range(FLAGS.num_epochs):
            print("{} Epoch number: {}".format(datetime.datetime.now(), epoch+1))
            step = 1

            # Epoch completed, start validation
            print("{} Start Test".format(datetime.datetime.now()))
            test_acc = 0.
            test_count = 0

            for _ in range(val_batches_per_epoch):
                batch_tx, batch_ty = val_preprocessor.next_batch(FLAGS.batch_size, 1)
                acc = sess.run(accuracy, feed_dict={x: batch_tx, y: batch_ty, dropout_keep_prob: 1.})
                test_acc += acc
                test_count += 1

            test_acc /= test_count
            print("{} Test Accuracy = {:.4f}".format(datetime.datetime.now(), test_acc))
            test_accuracy = test_acc

            # Reset the dataset pointers
            val_preprocessor.reset_pointer()

    # Write flags to txt
    flags_file_path = os.path.join(train_dir, 'flags.txt')
    flags_file = open(flags_file_path, 'w')
    flags_file.write('batch_size={}\n'.format(FLAGS.batch_size))
    flags_file.write('log_step={}\n'.format(FLAGS.log_step))
    flags_file.write('checkpoint_path={}\n'.format(FLAGS.ckpt_path))
    flags_file.write('test_accuracy={}'.format(test_accuracy))
    flags_file.close()


if __name__ == '__main__':
    tf.app.run()
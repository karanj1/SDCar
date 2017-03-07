import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
from keras.datasets import cifar10
from send_sms import send_sms
import time
from sklearn.utils import shuffle

# Lab: Cifar10 Aside : https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/e12c47b6-316e-4a0b-aae5-2f2c5fcd99f5/concepts/ce33f00b-c060-430c-b80d-b78cdd7373c6

# TODO: Split data into training and validation sets.
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# TODO: Split data into training and validation sets.
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=0)
# y_train.shape is 2d, (50000, 1). While Keras is smart enough to handle this
# it's a good idea to flatten the array.
y_train = y_train.reshape(-1)
y_val = y_val.reshape(-1)
y_test = y_test.reshape(-1)

nb_classes = 10

# TODO: Define placeholders and resize operation.
features = tf.placeholder(tf.float32, (None, 32, 32, 3))
labels = tf.placeholder(tf.int64, None)
resized = tf.image.resize_images(features, (227, 227))
epochs = 10
batch_size = 128
one_hot_y = tf.one_hot(labels, 10)      #https://www.tensorflow.org/api_docs/python/tf/one_hot

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)


# TODO: Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], nb_classes)  # use this shape for the weight matrix
fc8W = tf.Variable(tf.truncated_normal(shape, stddev=1e-2))
fc8b = tf.Variable(tf.zeros(nb_classes))
logits = tf.matmul(fc7, fc8W) + fc8b

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer()
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))  #https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmax.html
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# TODO: Train and evaluate the feature extraction model.

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, batch_size):
        batch_x, batch_y = X_data[offset:offset+batch_size], y_data[offset:offset+batch_size]
        accuracy = sess.run(accuracy_operation, feed_dict={features: batch_x, labels: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    for i in range(epochs):
        print("EPOCHS: " + format(i) + "\n")
        if i>0 and i%2==0:
            msg = "EPOCHS: " + format(i) + " and Validation accuracy(cifar 10) is: " + format(val_acc)
            send_sms(msg)
        # training
        X_train, y_train = shuffle(X_train, y_train)
        t0 = time.time()
        for offset in range(0, X_train.shape[0], batch_size):
            end = offset + batch_size
            sess.run(training_operation, feed_dict={features: X_train[offset:end], labels: y_train[offset:end]})

        val_acc = evaluate(X_val, y_val)
        print("Epoch", i+1)
        print("Time: %.3f seconds" % (time.time() - t0))
        #print("Validation Loss =", val_loss)
        print("Validation Accuracy =", val_acc)
        print("")

with tf.Session() as session:
    saver.restore(session, tf.train.latest_checkpoint('.'))
    
    test_acc = evaluate(X_test, y_test)
    print("Test Accuracy for Cifar10 = {:.3f}".format(test_acc))
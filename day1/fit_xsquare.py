import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime


def process(x):
    return x**2 + np.random.randn(x.size) * 1



epochs = 10000
batchsize = 50
learningrate = 0.001
N = 1000
trainsplit = 0.8
n_hidden = 500


trainsize = int(np.floor(trainsplit*N))
testsize = N - trainsize
num_batches = 50

x_ = tf.placeholder(tf.float32, [None, 1], name='x')
y_ = tf.placeholder(tf.float32, [None, 1], name='y')

x = np.random.rand(N)*10
y = process(x)

x = x
# y = y-50

x_train = x[:trainsize].reshape((trainsize,1))
y_train = y[:trainsize].reshape((trainsize,1))

x_test = x[trainsize:].reshape((testsize,1))
y_test = y[trainsize:].reshape((testsize,1))

W1 = tf.Variable(tf.random_normal([1, n_hidden], stddev=1/np.sqrt(n_hidden)), name='W1')
b1 = tf.Variable(tf.zeros([n_hidden]), name='b1')

# W2 = tf.Variable(tf.random_normal([5, 5], stddev=1/np.sqrt(n_hidden)), name='W2')
# b2 = tf.Variable(tf.zeros([5]), name='b2')

W3 = tf.Variable(tf.random_normal([n_hidden, 1], stddev=1/np.sqrt(n_hidden)), name='W3')
b3 = tf.Variable(tf.zeros([1]), name='b3')

W4 = tf.Variable(tf.random_normal([n_hidden, 1], stddev=1/np.sqrt(n_hidden)), name='W4')
b4 = tf.Variable(tf.zeros([1]), name='b4')

h1 = tf.nn.relu(tf.add(tf.matmul(x_, W1), b1))
# h2 = tf.nn.relu(tf.add(tf.matmul(h1, W2), b2))

mu      = tf.add(tf.matmul(h1, W3), b3)
sigma   = tf.nn.softplus(tf.add(tf.matmul(h1, W4), b4))

# mu      = output
# sigma   = tf.constant(1.0)
# sigma   = tf.nn.softplus(tf.Variable(tf.zeros([1])))

y_dist = tf.distributions.Normal(mu, sigma)

neg_log_likelihood = -1.0 * tf.reduce_mean(y_dist.log_prob(y_))


optimizer = tf.train.AdamOptimizer(learningrate)
train_op = optimizer.minimize(loss=neg_log_likelihood)

loss_summary = tf.summary.scalar(name="Loss curve train", tensor=neg_log_likelihood)
loss_train = tf.summary.scalar(name="Loss curve test", tensor=neg_log_likelihood)

init_op = tf.global_variables_initializer()

sess = tf.InteractiveSession()
sess.run(init_op)
log_path_train = 'logdir' + '/train_{}'.format(datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S"))
train_writer = tf.summary.FileWriter(log_path_train, sess.graph)
summaries_train = tf.summary.merge_all()
for epoch in range(epochs):
    total_loss = 0
    inds = np.array_split(np.random.permutation(len(x_train)), len(x_train)/batchsize)
    for batch, i in enumerate(inds):
        batch_x = x_train[i].reshape((batchsize,1))
        batch_y = y_train[i].reshape((batchsize,1))
        _, nll, summary_str = sess.run(
            [train_op, neg_log_likelihood, summaries_train],
            feed_dict={x_:batch_x, y_:batch_y})
        if batch % 10 == 0:
            train_writer.add_summary(summary_str, global_step=batch+epoch*num_batches)
            summary_test, mu_pred, sigma_pred = sess.run(
                [summaries_train, mu, sigma],
                feed_dict={x_:x_test, y_:y_test})
            train_writer.add_summary(summary_test, global_step=batch+epoch*num_batches)
    
    if epoch % 500 == 0:
        idx = np.argsort(x_test.ravel())
        plt.scatter(x_test[idx,0], y_test[idx,0], c='r')
        plt.fill_between(x_test[idx,0], mu_pred[idx,0]+3*sigma_pred[idx,0], mu_pred[idx,0]-3*sigma_pred[idx,0], alpha=0.3)
        plt.plot(x_test[idx,0], mu_pred[idx,0])
        plt.show()

    train_loss = sess.run(neg_log_likelihood, feed_dict={x_:x_train, y_:y_train})
    test_loss = sess.run(neg_log_likelihood, feed_dict={x_:x_test, y_:y_test})
    print("Epoch:", (epoch + 1), "Training loss =", "{:.3f}".format(train_loss))
    print("Epoch:", (epoch + 1), "Testing loss =", "{:.3f}".format(test_loss))
    


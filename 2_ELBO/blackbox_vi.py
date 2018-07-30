import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime


#################################################################
# SETUP PARAMETERS
epochs = 10000
learningrate = 0.01

# Create Dataset
x = tf.constant(2, dtype=tf.float32, name="x_hat")
sigma_eps = tf.constant(0.1, dtype=tf.float32, name="sigma_eps")

#################################################################
# CONSTRUCT MODEL
# q(z)
mu_z = tf.Variable(initial_value=0, trainable=True, dtype=tf.float32)
sigma_z = tf.Variable(initial_value=1, trainable=True, dtype=tf.float32)

q_z_dist = tf.distributions.Normal(mu_z, sigma_z, name="q_z_dist")
z = q_z_dist.sample(10, seed=None)

# likelihood
likelihood_dist = tf.distributions.Normal(
    tf.exp(z), sigma_eps, name="likelihood")
likelihood_plot_dist = tf.distributions.Normal(
    tf.exp(mu_z), sigma_eps, name="likelihood_plot")
# Monte carlo estimate of expectation
likelihood_loss = tf.reduce_mean(likelihood_dist.log_prob(x))

# prior p(z)
prior_mean = tf.constant(0, dtype=tf.float32)
prior_sigma = tf.constant(1, dtype=tf.float32)
prior_dist = tf.distributions.Normal(prior_mean, prior_sigma, name="prior")

# KL-loss
kl_loss = tf.distributions.kl_divergence(q_z_dist, prior_dist)

# ELBO
elbo = likelihood_loss - kl_loss
loss = -elbo
# Initialize optimizer
optimizer = tf.train.AdamOptimizer(learningrate)
train_op = optimizer.minimize(loss)

# Initialize Session and variables
init_op = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init_op)

#################################################################
# VISUALIZATION ONLY
elbo_summary = tf.summary.scalar(
    name="ELBO", tensor=elbo)
likelihood_summary = tf.summary.scalar(
    name="likelihood_loss", tensor=likelihood_loss)
kl_summary = tf.summary.scalar(
    name="KLD", tensor=kl_loss)
mean_summary = tf.summary.scalar(
    name="q(z)_mean", tensor=mu_z)
sigma_summary = tf.summary.scalar(
    name="q(z)_sigma", tensor=sigma_z)
log_path_train = 'logdir' + '/train_{}'.format(
    datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S"))
train_writer = tf.summary.FileWriter(log_path_train, sess.graph)
summaries_train = tf.summary.merge_all()
#################################################################

#################################################################
# TRAINING PROCESS
for epoch in range(epochs):
    _, elbo_calc, summary_str = sess.run(
        [
            train_op,
            elbo,
            summaries_train,
        ],
        feed_dict={})

    #################################################################
    # VISUALIZATION ONLY
    train_writer.add_summary(summary_str, global_step=epoch)
    if epoch % 10 == 0:
        quant_0_01, quant_0_99 = sess.run(
            [
                likelihood_plot_dist.quantile(0.01),
                likelihood_plot_dist.quantile(0.99)
            ])
        x_samples = tf.lin_space(quant_0_01, quant_0_99, 100)
        pdf_values = sess.run(likelihood_plot_dist.prob(x_samples))
        plt.cla()
        plt.plot(x_samples.eval(), pdf_values)
        plt.xlim(-4, 4)
        plt.ylim(0, 1)
        plt.xlabel('x')
        plt.ylabel('p(x|z)')
        plt.draw()
        plt.pause(0.1)

    # Print elbo each epoch
    print(
        "Epoch:", (epoch + 1),
        "ELBO =", "{:.3f}".format(elbo_calc))

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime


#################################################################
# SETUP PARAMETERS
epochs = 1500
learningrate = 0.01
training_name = input("Please enter training_name: ")

# Create Dataset
x = tf.constant(2, dtype=tf.float32, name="x_hat")
sigma_eps = tf.constant(0.1, dtype=tf.float32, name="sigma_eps")

#################################################################
# CONSTRUCT MODEL
# q(z)
mu_z = tf.Variable(initial_value=2, trainable=True, dtype=tf.float32)
sigma_z_var = tf.Variable(initial_value=1, trainable=True, dtype=tf.float32)
sigma_z = 0.5 * tf.sqrt(tf.square(sigma_z_var) + 0.001) + sigma_z_var * 0.5

q_z_dist = tf.distributions.Normal(mu_z, sigma_z, name="q_z_dist")
z = q_z_dist.sample(10, seed=None)

# likelihood
likelihood_dist = tf.distributions.Normal(
    tf.sin(4*z), sigma_eps, name="likelihood")
# Monte carlo estimate of expectation
likelihood_loss = tf.reduce_mean(likelihood_dist.log_prob(x))

# prior p(z)
prior_mean = tf.constant(3, dtype=tf.float32)
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
if len(training_name) == 0:
    log_path_train = 'logdir' + '/train_{}'.format(
        datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S"))
else:
    log_path_train = 'logdir' + '/train_{}'.format(
        training_name)
train_writer = tf.summary.FileWriter(log_path_train, sess.graph)
summaries_train = tf.summary.merge_all()
#################################################################

#################################################################
# APPROXIMATE POSTERIOR
z_samples_AP = tf.lin_space(0.0, 7.0, 1000)
likelihood_AP = tf.distributions.Normal(tf.sin(4*z_samples_AP), sigma_eps)
joint_AP = prior_dist.prob(z_samples_AP) * likelihood_AP.prob(x)
dz = z_samples_AP[1].eval() - z_samples_AP[0].eval()
posterior_AP = joint_AP / tf.reduce_sum(joint_AP) / dz

ap, sjap, jap = sess.run([posterior_AP, tf.reduce_sum(joint_AP), joint_AP])

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
                q_z_dist.quantile(0.001),
                q_z_dist.quantile(0.999),
            ])
        z_samples = tf.lin_space(quant_0_01, quant_0_99, 100)
        pdf_values = sess.run(q_z_dist.prob(z_samples))
        plt.cla()
        plt.plot(z_samples.eval(), pdf_values, label='q(z)')
        plt.plot(z_samples_AP.eval(), ap, label='p(z|x)')
        plt.xlim(0, 7)
        plt.ylim(0, 10)
        plt.xlabel('x')
        plt.legend()
        plt.draw()
        plt.pause(0.1)

        # Print elbo every 10th epoch
        print(
            "Epoch:", (epoch + 1),
            "ELBO =", "{:.3f}".format(elbo_calc))

plt.pause(-1)
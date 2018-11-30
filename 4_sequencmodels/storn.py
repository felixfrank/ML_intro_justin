import tensorflow as tf
import numpy as np
import datetime
import pickle
import os


#########################################
# Parameters
memory_size = 100
latent_size = 1

class Model(object):
    """Build simple storn model"""
    def __init__(self, input_size, memory_size=100, latent_size=1, init_scale=0.1):
        self.input_size = input_size
        self.memory_size = memory_size
        self.latent_size = latent_size
        self.init_scale = init_scale

        self._obs = tf.placeholder(tf.float32, shape=[None, None, input_size])
        self._mask = tf.placeholder(tf.float32, shape=[None, None])
        self._init_state = tf.placeholder(tf.float32, shape=[None, memory_size])
        initializer = tf.random_uniform_initializer(-init_scale, init_scale)
        with tf.variable_scope("model", initializer=initializer):
            self._fstates, self._bstates = self._compute_istates()
            self._istates = tf.concat((self.fstates, self.bstates), -1)
            self._q = self._compute_q()
            self._z = self.q.sample()
            self._gstates = self._compute_gstates(self.z)
            self._posterior = self._compute_posterior(self.gstates)
            self._loss = self.compute_loss()

    def _rnn_step(self, h_prev, x):
        in_size = x.get_shape().as_list()[-1]
        Wh = tf.get_variable("Wh", shape=[self.memory_size, self.memory_size])
        Wx = tf.get_variable("Wx", shape=[in_size, self.memory_size])
        b = tf.get_variable("b", shape=[self.memory_size])
        h = tf.nn.sigmoid(tf.matmul(h_prev, Wh) + tf.matmul(x, Wx) + b)
        return h

    def _linear_layer(self, x, in_size, out_size):
        W = tf.get_variable("W", shape=[in_size, out_size])
        b = tf.get_variable("b", shape=[out_size])
        return tf.einsum("ijk,kl->ijl", x, W) + b

    def _compute_q(self):
        with tf.variable_scope("q"):
            statistics = self._linear_layer(self.istates, 2*self.memory_size, 2*self.latent_size)
        loc, scale = tf.split(value=statistics, axis=-1, num_or_size_splits=2)
        scale = tf.square(scale)
        return tf.distributions.Normal(loc, scale)

    def _compute_posterior(self, gstates):
        with tf.variable_scope("likelihood"):
            statistics = self._linear_layer(gstates, self.memory_size, self.input_size)
        self._bpval = statistics
        return tf.distributions.Bernoulli(logits=statistics)

    def _compute_istates(self):
        with tf.variable_scope("inference_forward"):
            fstates = tf.scan(self._rnn_step, self.obs, initializer=self.init_state, name="states")
        with tf.variable_scope("inference_backward"):
            bstates = tf.scan(self._rnn_step, self.obs, initializer=self.init_state, name="states", reverse=True)
        return fstates, bstates

    def _compute_gstates(self, z):
        with tf.variable_scope("generative_forward"):
            padv = tf.constant([[1, 0], [0, 0], [0, 0]])
            x = tf.pad(self.obs, padv, "CONSTANT")
            self._x_z = tf.concat((x[:-1], z), axis=-1)
            gstates = tf.scan(self._rnn_step, self.x_z, initializer=self.init_state, name="states")
        return gstates

    def compute_loss(self):
        prior = tf.distributions.Normal(0., 1.)
        self._kl = tf.distributions.kl_divergence(self.q, prior)
        self._loglikelihood = tf.reduce_sum(self.posterior.prob(self.obs), axis=-1)
        self._sample_loss = (-self.loglikelihood + tf.reduce_sum(self.kl, axis=-1)) * self.mask
        return tf.reduce_sum(self.sample_loss)

    def _predict(self, h_x_prev, z):
        with tf.variable_scope("model", reuse=True):
            with tf.variable_scope("generative_forward", reuse=True):
                h_prev, x_prev = tf.split(h_x_prev, axis=0, num_or_size_splits=[self.memory_size, self.input_size])
                x_z = tf.concat((x_prev, z), axis=0)
                Wh = tf.get_variable("Wh", shape=[self.memory_size, self.memory_size])
                Wx = tf.get_variable("Wx", shape=[self.input_size + self.latent_size, self.memory_size])
                b = tf.get_variable("b", shape=[self.memory_size])
                h = tf.nn.sigmoid(tf.einsum("i,ij->j", h_prev, Wh) + tf.einsum("i,ij->j", x_z, Wx) + b)
                gstates = tf.expand_dims(h, axis=0)
                gstates = tf.expand_dims(gstates, axis=0)
            posterior = self._compute_posterior(gstates)
            x = tf.cast(tf.squeeze(posterior.sample()), tf.float32)
        return tf.concat(values=(h, x), axis=0)

    def compose(self, seqlen):
        prior = tf.distributions.Normal(np.zeros(self.latent_size, dtype=np.float32), np.ones(self.latent_size, dtype=np.float32))
        z = prior.sample(seqlen)
        h_x = tf.scan(self._predict, z, initializer=np.zeros(self.memory_size + self.input_size, dtype=np.float32))
        h, x = tf.split(axis=1, value=h_x, num_or_size_splits=[self.memory_size, self.input_size])
        return x

    @property
    def obs(self):
        return self._obs

    @property
    def fstates(self):
        return self._fstates

    @property
    def bstates(self):
        return self._bstates

    @property
    def istates(self):
        return self._istates

    @property
    def gstates(self):
        return self._gstates

    @property
    def q(self):
        return self._q

    @property
    def kl(self):
        return self._kl

    @property
    def z(self):
        return self._z

    @property
    def posterior(self):
        return self._posterior

    @property
    def x_z(self):
        return self._x_z

    @property
    def mask(self):
        return self._mask

    @property
    def loss(self):
        return self._loss

    @property
    def sample_loss(self):
        return self._sample_loss

    @property
    def loglikelihood(self):
        return self._loglikelihood

    @property
    def bpval(self):
        return self._bpval

    @property
    def predictions(self):
        return self._predictions

    @property
    def init_state(self):
        return self._init_state
    
# data = [[[0, 0, 0, 0, 0], [0, 1, 0, 0, 1], [0, 1, 1, 0, 1], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0], [1, 0, 0, 0, 0]], [[0, 0, 0, 1, 0], [0, 0, 0, 0, 1], [1, 1, 1, 0, 1], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0], [1, 0, 0, 0, 1]]]
# data = np.array(data)
# data = np.swapaxes(data, 0, 1)
# mask = np.ones(data.shape[:-1], dtype=np.bool)
# seqlen = data.shape[0]
# batchsize = data.shape[1]
# obs_size = data.shape[2]
# init = np.zeros((batchsize, memory_size))

file = open("data/Piano-midi.de.pickle", "rb")
data = pickle.load(file)
from utilities import data_to_dataset
train_data, train_mask = data_to_dataset(data["train"])
train_batchsize = train_data.shape[0]
train_seqlen = train_data.shape[1]
train_obs_size = train_data.shape[2]
train_init = np.zeros((train_batchsize, memory_size))
valid_data, valid_mask = data_to_dataset(data["valid"])
valid_seqlen = valid_data.shape[0]
valid_batchsize = valid_data.shape[1]
valid_obs_size = valid_data.shape[2]
valid_init = np.zeros((valid_batchsize, memory_size))
# test_data, test_mask = data_to_dataset(data["test"])
data_placeholder = tf.placeholder(tf.float32, shape=(train_batchsize, train_seqlen, train_obs_size))
mask_placeholder = tf.placeholder(tf.float32, shape=(train_batchsize, train_seqlen))
init_placeholder = tf.placeholder(tf.float32, shape=(train_batchsize, memory_size))
data = tf.data.Dataset().from_tensor_slices(data_placeholder).batch(train_batchsize)
mask = tf.data.Dataset().from_tensor_slices(mask_placeholder).batch(train_batchsize)
init = tf.data.Dataset().from_tensor_slices(init_placeholder).batch(train_batchsize)

data_iterator = data.shuffle(train_batchsize).make_initializable_iterator()
mask_iterator = mask.shuffle(train_batchsize).make_initializable_iterator()
init_iterator = init.shuffle(train_batchsize).make_initializable_iterator()

data_batch_handle = tf.transpose(data_iterator.get_next(), perm=[1, 0, 2])
mask_batch_handle = tf.transpose(mask_iterator.get_next(), perm=[1, 0])
init_batch_handle = init_iterator.get_next()

model = Model(train_obs_size, memory_size, latent_size)
optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(model.loss)

train_loss_summary = tf.summary.scalar(name="training loss", tensor=model.loss)
valid_loss_summary = tf.summary.scalar(name="validation loss", tensor=model.loss)

current_time = datetime.datetime.utcnow().strftime("%Y_%m_%d_%H_%M_%S")
log_path_train = 'logdir' + '/train_{}'.format(current_time)
os.mkdir("generated_music/pieces_{}".format(current_time))
saver = tf.train.Saver()

sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
with tf.Session(config=sess_config) as sess:
    train_writer = tf.summary.FileWriter(log_path_train, sess.graph)
    # summaries_train = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())
    for epoch in range(1000):
        sess.run((data_iterator.initializer, mask_iterator.initializer, init_iterator.initializer), {data_placeholder: train_data, mask_placeholder: train_mask, init_placeholder: train_init})
        try:
            # Go through the entire dataset
            while True:
                data_batch, mask_batch, init_batch = sess.run((data_batch_handle, mask_batch_handle, init_batch_handle))
                # , {data_placeholder: train_data, mask_placeholder: train_mask, init_placeholder: train_init}) 
                _, loss, train_summary = sess.run(
                    [train_op, model.loss, train_loss_summary],
                    {model.obs: data_batch,
                    model.mask: mask_batch,
                    model.init_state: init_batch})
                valid_summary = sess.run(
                    valid_loss_summary,
                    {model.obs: valid_data,
                    model.mask: valid_mask,
                    model.init_state: valid_init})
                
        except tf.errors.OutOfRangeError:
            print('End of Epoch.')
        
        train_writer.add_summary(train_summary, global_step=epoch)
        train_writer.add_summary(valid_summary, global_step=epoch)
        train_writer.flush()

        print(
            "Epoch:", (epoch + 1),
            "LOSS =", "{:.3f}".format(loss))

        if epoch % 50 == 0:
            music = sess.run(model.compose(300))
            file = open("generated_music/pieces_{}/epoch_{}.pickle".format(current_time, epoch), "wb")
            pickle.dump(music, file)
            file.close()
            save_path = saver.save(sess, "generated_music/pieces_{}/epoch_{}.ckpt".format(current_time, epoch))
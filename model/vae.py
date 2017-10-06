import time, sys, os.path, os
sys.path.append("./")
import tensorflow as tf
import numpy as np
from tensorflow.contrib import losses
from tensorflow.contrib import slim
#from tensorflow.contrib import losses
from util.image import nchw_to_nhwc
from analyzer import read_whole_features, SPEAKERS, pw2wav, Tanhize
from util.wrapper import get_default_output, convert_f0, nh_to_nchw
from util.layers import (GaussianKLD, GaussianLogDensity, GaussianSampleLayer,
                         Layernorm, conv2d_nchw_layernorm, lrelu)
# from model.wgan import GradientPenaltyWGAN

class ConvVAE(object):
    def __init__(self, arch, is_training=False):
        '''
        Variational auto-encoder implemented in 2D convolutional neural nets
        Input:
            `arch`: network architecture (`dict`)
            `is_training`: (unused now) it was kept for historical reasons (for `BatchNorm`)
        '''
        self.arch = arch
        self._sanity_check()
        self.is_training = is_training

        with tf.name_scope('SpeakerRepr'):
            self.y_emb = self._l2_regularized_embedding(
                self.arch['y_dim'],
                self.arch['z_dim'],
                'y_embedding')

        self._generate = tf.make_template(
            'Generator',
            self._generator)

        self._encode = tf.make_template(
            'Encoder',
            self._encoder)

        self.generate = self.decode  # for VAE-GAN extension


    def _sanity_check(self):
        for net in ['encoder', 'generator']:
            assert len(self.arch[net]['output']) == len(self.arch[net]['kernel']) == len(self.arch[net]['stride'])


    def _unit_embedding(self, n_class, h_dim, scope_name, var_name='y_emb'):
        with tf.variable_scope(scope_name):
            embeddings = tf.get_variable(
                name=var_name,
                shape=[n_class, h_dim])
            embeddings = tf.nn.l2_normalize(embeddings, dim=-1, name=var_name+'normalized')
        return embeddings


    def _merge(self, var_list, fan_out, l2_reg=1e-6):
        x = 0.
        with slim.arg_scope(
            [slim.fully_connected],
            num_outputs=fan_out,
            weights_regularizer=slim.l2_regularizer(l2_reg),
            normalizer_fn=None,
            activation_fn=None):
            for var in var_list:
                x = x + slim.fully_connected(var)
        return slim.bias_add(x)


    def _l2_regularized_embedding(self, n_class, h_dim, scope_name, var_name='y_emb'):
        with tf.variable_scope(scope_name):
            embeddings = tf.get_variable(
                name=var_name,
                shape=[n_class, h_dim],
                regularizer=slim.l2_regularizer(1e-6))
        return embeddings

    def _encoder(self, x, is_training=None):
        net = self.arch['encoder']
        print "x shape:", x.get_shape().as_list()
        #time.sleep(100)
        for i, (o, k, s) in enumerate(zip(net['output'], net['kernel'], net['stride'])):
            x = conv2d_nchw_layernorm(
                x, o, k, s, lrelu,
                name='Conv2d-{}'.format(i)
            )
            print("ConvT-LN{}, shape:".format(i), x.get_shape().as_list())
        print "x shape:", x.get_shape().as_list()
        x = slim.flatten(x)
        print "x shape:", x.get_shape().as_list()
        #time.sleep(100)
        z_mu = tf.layers.dense(x, self.arch['z_dim'])
        z_lv = tf.layers.dense(x, self.arch['z_dim'])
        print "z_mu:",z_mu.get_shape().as_list()
        #time.sleep(100)
        return z_mu, z_lv

    def _generator(self, z, y, is_training=None):
        net = self.arch['generator']
        h, w, c = net['hwc']

        if y is not None:
            y = tf.nn.embedding_lookup(self.y_emb, y)
            x = self._merge([z, y], h * w * c)
        else:
            x = z
        print "generate x shape:", x.get_shape().as_list()
        x = tf.reshape(x, [-1, c, h, w])  # channel first
        print "generate x reshape:", x.get_shape().as_list()
        for i, (o, k, s) in enumerate(zip(net['output'], net['kernel'], net['stride'])):
            x = tf.layers.conv2d_transpose(x, o, k, s,
                padding='same',
                data_format='channels_first',
            )
            if i < len(net['output']) -1:
                x = Layernorm(x, [1, 2, 3], 'ConvT-LN{}'.format(i))
                x = lrelu(x)
            print("ConvT-LN{}, shape:".format(i), x.get_shape().as_list())
        print "generate x shape:", x.get_shape().as_list()
        return x


    def loss(self, x, y):
        print "x-first shape:", x.get_shape().as_list()
        with tf.name_scope('loss'):
            z_mu, z_lv = self._encode(x)
            z = GaussianSampleLayer(z_mu, z_lv)
            xh = self._generate(z, y)

            D_KL = tf.reduce_mean(
                GaussianKLD(
                    slim.flatten(z_mu),
                    slim.flatten(z_lv),
                    slim.flatten(tf.zeros_like(z_mu)),
                    slim.flatten(tf.zeros_like(z_lv)),
                )
            )
            logPx = tf.reduce_mean(
                GaussianLogDensity(
                    slim.flatten(x),
                    slim.flatten(xh),
                    tf.zeros_like(slim.flatten(xh))),
            )

        loss = dict()
        loss['G'] = - logPx + D_KL
        loss['D_KL'] = D_KL
        loss['logP'] = logPx

        tf.summary.scalar('KL-div', D_KL)
        tf.summary.scalar('logPx', logPx)

        tf.summary.histogram('xh', xh)
        tf.summary.histogram('x', x)
        return loss

    def encode(self, x):
        z_mu, _ = self._encode(x)
        return z_mu

    def decode(self, z, y):
        xh = self._generate(z, y)
        return nchw_to_nhwc(xh)

class VAWGAN(object):
    def __init__(self, arch, args, is_training, reuse):
        self.arch = arch
        self.args = args
        self.reuse= reuse
        self.is_training = is_training
        with tf.name_scope("SpeakerRepr"):
            self.y_emb = self._l2_regularizer_embedding(
            self.arch['y_dim'],
            self.arch['z_dim'],
            'y_embedding')


    def _l2_regularizer_embedding(self, n_class, h_dim, scope_name, var_name = "y_emb"):
        with tf.variable_scope(scope_name):
            embeddings = tf.get_variable(
            name = scope_name, shape = [n_class, h_dim])
            embeddings = tf.nn.l2_normalize(embeddings, dim=-1, name=var_name+'normalized')
        return embeddings

    def encoder(self, x, is_training, reuse=False):
        n_layer = len(self.arch['encoder']['output'])
        subnet = self.arch['encoder']
        h, w, c = self.arch['hwc']
        x = tf.reshape(x, [-1, c, h, w])
        print "encoder shape:", x.get_shape().as_list()
        with tf.variable_scope("encoder", reuse=reuse):
            with slim.arg_scope(
                [slim.batch_norm],
                scale=True,
                data_format = "NCHW",
                decay=0.99, epsilon=1e-5,
                is_training=is_training,
                trainable=True,
                scope = 'e-BN'):
                with slim.arg_scope(
                        [slim.conv2d],
                        padding = "same",
                        data_format="NCHW",
                        weights_regularizer=slim.l2_regularizer(subnet['l2-reg']),
                        normalizer_fn=slim.batch_norm,
                        trainable = True,
                        activation_fn=tf.nn.relu):

                    for i in range(n_layer):
                        x = slim.conv2d(
                            x,
                            subnet['output'][i],
                            subnet['kernel'][i],
                            subnet['stride'][i])
                        print "encoder x shape:", x.get_shape().as_list()
            x = slim.flatten(x)

            with slim.arg_scope(
                [slim.fully_connected],
                num_outputs=self.arch['z_dim'],
                weights_regularizer=slim.l2_regularizer(subnet['l2-reg']),
                trainable = True,
                normalizer_fn=None,
                activation_fn=None):
                z_mu = slim.fully_connected(x)
                z_lv = slim.fully_connected(x)
        print "z_mu shape:", z_mu.get_shape().as_list()
        return z_mu, z_lv

    def _merge(self, var_list, fan_out, l2_reg=1e-6):
        x = 0.
        with slim.arg_scope(
            [slim.fully_connected],
            num_outputs=fan_out,
            weights_regularizer=slim.l2_regularizer(l2_reg),
            normalizer_fn=slim.batch_norm,
            trainable=True,
            activation_fn=tf.nn.relu):
            for var in var_list:
                x = x + slim.fully_connected(var)

        return slim.bias_add(x)

    def generator(self, z, y, is_training, reuse=False):
        #tf.get_variable_scope().reuse_variables()

        n_layer = len(self.arch['generator']['output'])
        subnet = self.arch['generator']
        h, w, c = subnet['hwc']
        with tf.variable_scope("generator", reuse=reuse):
            y = tf.nn.embedding_lookup(self.y_emb, y)
            x = self._merge([z, y], h * w * c)
            print ("generator x shape", x.get_shape().as_list())
            x = tf.reshape(x, [-1, c, h, w])
            print("generator x shape:", x.get_shape().as_list())
            with slim.arg_scope(
                [slim.batch_norm],
                scale=True,
                trainable = True,
                data_format = "NCHW",
                decay=0.99, epsilon=1e-5, #decay = 0.997
                is_training=is_training,
                scope='g-BN'):
                '''
                x = slim.fully_connected(
                    x,
                    h * w * c,
                    normalizer_fn=slim.batch_norm,
                    trainable=True,
                    activation_fn=tf.nn.relu)
                x = tf.reshape(x, [-1, c, h, w])
                '''
                with slim.arg_scope(
                        [slim.conv2d_transpose],
                        padding = "same",
                        data_format = 'NCHW',
                        trainable = True,
                        weights_regularizer=slim.l2_regularizer(subnet['l2-reg']),
                        normalizer_fn=slim.batch_norm,
                        activation_fn=tf.nn.relu):

                    for i in range(n_layer -1):
                        x = slim.conv2d_transpose(
                            x,
                            subnet['output'][i],
                            subnet['kernel'][i],
                            subnet['stride'][i])
                        print "conv2d_transpose x shape:",x.get_shape().as_list()
                    # Don't apply BN for the last layer of G
                    x = slim.conv2d_transpose(
                        x,
                        subnet['output'][-1],
                        subnet['kernel'][-1],
                        subnet['stride'][-1],
                        normalizer_fn=None,
                        trainable=True,
                        activation_fn=tf.nn.tanh
                        )
                print "conv2d_transpose x shape:",x.get_shape().as_list()
        print "generate x shape:", x.get_shape().as_list()
        return x

    def nchw_to_nhwc(self, x):
        x= tf.transpose(x, [0, 2, 3, 1])
        return x

    def nhwc_to_nchw(self, x):
        return tf.transpose(x, [0, 3, 1, 2])

    def discriminator(self, x, is_training):

        n_layer = len(self.arch['discriminator']['output'])
        h, w, c = self.arch['hwc']
        subnet = self.arch['discriminator']
        feature = list()
        x = self.nhwc_to_nchw(x)
        print("discriminator x shape:", x.get_shape().as_list())
        with slim.arg_scope(
            [slim.batch_norm],
            scale=True,
            trainable = True,
            data_format="NCHW",
            decay=0.99, epsilon=1e-5,
            is_training=is_training,

            scope = 'd-BN'):
            with slim.arg_scope(
                    [slim.conv2d],
                    weights_regularizer=slim.l2_regularizer(subnet['l2-reg']),
                    normalizer_fn=slim.batch_norm,
                    activation_fn=tf.nn.relu,
                    trainable = True,
                    data_format = "NCHW"
                    ):
                x = slim.conv2d(
                    x,
                    subnet['output'][0],
                    subnet['kernel'][0],
                    subnet['stride'][0],
                    normalizer_fn=None
                    )
                feature.append(x)
                for i in range(1, n_layer):
                    x = slim.conv2d(
                        x,
                        subnet['output'][i],
                        subnet['kernel'][i],
                        subnet['stride'][i],
                        )
                    feature.append(x)

        print("x shape:", x.get_shape().as_list())
        x = slim.flatten(x)
        h = slim.flatten(feature[subnet['feature_layer'] - 1])
        if h.get_shape().as_list()[-1] == 2736:
            h = tf.layers.dense(h, 16, trainable=True)
        print("h shape:", h.get_shape().as_list())
        print("x flattend shape:", x.get_shape().as_list())
        x = slim.fully_connected(x, 1, activation_fn = None, trainable=True)
        return x, h  # no explicit `sigmoid`

    def loss(self, x, y, is_training=True):
        with tf.name_scope('loss'):
            with tf.variable_scope("encoder") as scope:  #specify variable_scope
                z_mu, z_lv = self.encoder(x, is_training)       #so that to collect trainable
                z = GaussianSampleLayer(z_mu, z_lv) # variables
                D_KL = tf.reduce_mean(
                    GaussianKLD(
                        slim.flatten(z_mu),
                        slim.flatten(z_lv),
                        slim.flatten(tf.zeros_like(z_mu)),
                        slim.flatten(tf.zeros_like(z_lv)),
                    )
                )
            with tf.variable_scope("generator") as scope:
                xh = self.generator(z, y, is_training)
                print("xh shape:", xh.get_shape().as_list())
                xh = self.nchw_to_nhwc(xh)
                print("xh shape:", xh.get_shape().as_list())
            with tf.variable_scope("discriminator") as scope:
                disc_real, x_through_d = self.discriminator(x, is_training)
                print("disc_real shape:", disc_real.get_shape().as_list())
                print("x_through_d:", x_through_d.get_shape().as_list())
                disc_fake, xh_through_d = self.discriminator(xh, is_training)
                logPx = -tf.reduce_mean(
                    GaussianLogDensity(
                        x_through_d,
                        xh_through_d,
                        tf.zeros_like(xh_through_d)),
                )


        loss = dict()
        loss['D_KL'] = D_KL
        loss['logP'] = logPx

        batch_size = self.arch['training']['batch_size']

        #disc_real_loss = tf.losses.sigmoid_cross_entropy(disc_real, tf.ones([batch_size, 1]))
        #disc_fake_loss = tf.losses.sigmoid_cross_entropy(disc_fake, tf.fill([batch_size, 1], -1.0))
        gen_loss = -tf.reduce_mean(disc_fake)
        disc_loss = tf.reduce_mean(disc_fake - disc_real)

        alpha = tf.random_uniform(
            shape=[batch_size, 513,1,1],
            minval=0.,
            maxval=1.
        )

        #gradient penalty
        print("before gradient x shape:", x.get_shape().as_list())
        differences = xh - x
        interpolates = x + (alpha*differences)
        print("interpolates shape:", interpolates.get_shape().as_list())
        pred, inter_h = self.discriminator(interpolates, is_training)
        print("pred shape:", pred.get_shape().as_list())
        gradients = tf.gradients(pred, [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1,2]))
        gradient_penalty = tf.reduce_mean((slopes-1.)**2)
        disc_loss += self.arch['LAMBDA']*gradient_penalty

        #d_loss = disc_real_loss + disc_fake_loss
        #g_loss = tf.losses.sigmoid_cross_entropy(disc_fake, tf.ones([batch_size, 1]))
        g_loss = gen_loss
        d_loss = disc_loss

        loss['l_G'] = g_loss
        loss['l_D'] = d_loss
        loss['l_E'] = D_KL + logPx
        loss['G'] = D_KL + logPx + 50.*d_loss
        return loss

    def sample(self):
        with tf.name_scope("sample"):
            normalizer = Tanhize(
                xmax=np.fromfile('./etc/xmax.npf'),
                xmin=np.fromfile('./etc/xmin.npf'),
            )
            FEAT_DIM = 1029
            SP_DIM = 513
            self.reues=True
            #features = read_whole_features(self.args.file_pattern.format(self.args.src))
            files = "./dataset/vcc2016/bin/Testing Set/SF1/200005.bin"
            #filename_queue = tf.train.string_input_producer(files, num_epochs=1)
            #reader = tf.WholeFileReader()
            #key, value = reader.read(filename_queue)
            key = tf.cast(os.path.split(files)[-1].split('.')[0], tf.string)
            with open(files, 'rb') as f:
                value = f.read()
            value = tf.decode_raw(value, tf.float32)
            value = tf.reshape(value, [-1, FEAT_DIM])
            features = dict()
            features['sp']= value[:, :SP_DIM]
            features['ap']= value[:, SP_DIM : 2*SP_DIM]
            features['f0']= value[:, SP_DIM * 2]
            features['en']= value[:, SP_DIM * 2 + 1]
            features['speaker']= tf.cast(value[:, SP_DIM * 2 + 2], tf.int64)
            features['filename']= key
            #x = normalizer.forward_process(features['sp'])
            x = tf.clip_by_value(features['sp'], 0., 1.)*2. -1.
            x = nh_to_nchw(x)
            #y_t_id = tf.placeholder(dtype=tf.int64, shape=[1,])
            #y_t = y_t_id * tf.ones(shape=[tf.shape(x)[0],], dtype=tf.int64)
            y_t = SPEAKERS.index(self.args.trg)*tf.ones(shape=[tf.shape(x)[0],], dtype=tf.int64)
            with tf.variable_scope("encoder", reuse=True):
                z, _ = self.encoder(x, False, True)
            with tf.variable_scope("generator", reuse=True):
                x_t = self.generator(z, y_t, False, True)
            x_t = tf.transpose(x_t, [0, 2, 3, 1])
            print "x_t shape:", x_t.get_shape().as_list()
            x_t = tf.squeeze(x_t)
            x_t = normalizer.backward_process(x_t)

            f0_s = features['f0']
            f0_t = convert_f0(f0_s, self.args.src, self.args.trg)
        sample=dict()
        sample['features'] = features
        sample['x_t'] = x_t
        sample['f0_t'] = f0_t
        #sample['y_t'] = y_t_id
        return sample

import time, sys
sys.path.append("./")
import tensorflow as tf
from tensorflow.contrib import losses
from tensorflow.contrib import slim
#from tensorflow.contrib import losses
from util.image import nchw_to_nhwc
from util.ops import *
from analyzer import read_whole_features, SPEAKERS, pw2wav, Tanhize
from util.wrapper import get_default_output, convert_f0, nh_to_nchw
from util.layers import (GaussianKLD, GaussianLogDensity, GaussianSampleLayer,
                         Layernorm, conv2d_nchw_layernorm, lrelu)
# from model.wgan import GradientPenaltyWGAN

class VAWGAN(object):
    def __init__(self, arch, args, is_training, reuse=False):
        self.arch = arch
        self.args = args
        self.is_training = is_training
        self.reuse = reuse
        with tf.name_scope("SpeakerRepr"):
            self.y_emb = self._l2_regularizer_embedding(
            self.arch['y_dim'],
            self.arch['z_dim'],
            'y_embedding')

    def _l2_regularizer_embedding(self, n_class, h_dim, scope_name, var_name = "y_emb"):
        with tf.variable_scope(scope_name, reuse=None):
            embeddings = tf.get_variable(
            name = var_name, shape = [n_class, h_dim],regularizer=slim.l2_regularizer(1e-6))
            embeddings = tf.nn.l2_normalize(embeddings, dim=-1, name=var_name+'normalized')
        return embeddings

    def encoder(self, x, is_training):
        n_layer = len(self.arch['encoder']['output'])
        subnet = self.arch['encoder']
        h, w, c = self.arch['hwc']
        x = tf.reshape(x, [-1, c, h, w])
        print "encoder shape:", x.get_shape().as_list()
        with tf.variable_scope("encoder",reuse = self.reuse) as scope:
            for i in range(n_layer):
                x = conv2d(x, subnet['output'][i],
                subnet['kernel'][i],
                subnet['stride'][i],
                name="conv2d-"+str(i),
                norm = "batch_norm" )
                x = tf.maximum(0.3*x, x)
            x = slim.flatten(x)
            x = linear(x, self.arch['z_dim'], scope= "en_ful")
            z_lv = x
        print "x shape:", x.get_shape().as_list()
        return x, z_lv

    def _merge(self, var_list, fan_out, reuse =False, l2_reg=1e-6):
        x = 0.
        with tf.variable_scope("_merge", reuse=reuse) as scope:
            x = linear(var_list[0], fan_out, name="linear-0")
            x = linear(var_list[1], fan_out, name="linear-1")
            print "merged x shape:", x.get_shape().as_list()
            biases = tf.get_variable('merge_biases', [fan_out], initializer=tf.constant_initializer(0.0))
            x = x + biases
            return x

    def generator(self, z, y, is_training):
        #tf.get_variable_scope().reuse_variables()

        n_layer = len(self.arch['generator']['output'])
        subnet = self.arch['generator']
        h, w, c = subnet['hwc']
        with tf.variable_scope("generator", reuse=self.reuse) as scope:
            y = tf.nn.embedding_lookup(self.y_emb, y, self.reuse)
            x = self._merge([z, y], h * w * c, reuse=self.reuse)
            print ("generator x shape", x.get_shape().as_list())
            x = tf.reshape(x, [-1, c, h, w])
            print ("x reshape", x.get_shape().as_list())
            for i in range(n_layer - 1):
                x = deconv2d(x, subnet['output'][i],
                subnet['kernel'][i],
                subnet['stride'][i],
                name="deconv2d-"+str(i),
                norm = "batch_norm" ,
                reuse=self.reuse)
                x = tf.maximum(0.3*x, x)
            x = deconv2d(x, subnet['output'][3], subnet['kernel'][3],
            subnet['stride'][3], reuse=self.reuse)
            x = tf.nn.tanh(x)
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
        with tf.variable_scope("discriminator", reuse=self.reuse) as scope:
            x = conv2d(x, subnet['output'][0], subnet['kernel'][0],
            subnet['stride'][0], name='disconv2d-0', reuse=self.reuse)
            x = tf.maximum(0.3*x, x)
            feature.append(x)
            for i in range(1, n_layer):
                x = conv2d(x, subnet['output'][i],
                subnet['kernel'][i],
                subnet['stride'][i],
                name="disconv2d-"+str(i),
                norm = "batch_norm", reuse =self.reuse)
                x = tf.maximum(0.3*x, x)
                feature.append(x)
            x = slim.flatten(x)
            h = slim.flatten(feature[subnet['feature_layer'] - 1])
            if h.get_shape().as_list()[-1] == 2736:
                h = linear(h, 16, name= 'd_layer', reuse=self.reuse)
            x = linear(x, 1, name="d_ful", reuse=self.reuse)
            self.reuse=True
        return x, h


    def loss(self, x, y):
        with tf.name_scope('loss'):
            with tf.variable_scope("encoder") as scope:  #specify variable_scope
                z_mu, z_lv = self.encoder(x, self.is_training)       #so that to collect trainable
                z = GaussianSampleLayer(z_mu, z_lv) # variables
            with tf.variable_scope("generator") as scope:
                xh = self.generator(z, y, self.is_training)
                print("xh shape:", xh.get_shape().as_list())
                #xh = self.nchw_to_nhwc(xh)
                print("xh shape:", xh.get_shape().as_list())
            with tf.variable_scope("discriminator") as scope:
                #x = nchw_to_nhwc(x)

                disc_real, x_through_d = self.discriminator(x, self.is_training)
                print("disc_real shape:", disc_real.get_shape().as_list())
                print("x_through_d:", x_through_d.get_shape().as_list())
                disc_fake, xh_through_d = self.discriminator(xh, self.is_training)

            D_KL = tf.reduce_mean(
                GaussianKLD(
                    slim.flatten(z_mu),
                    slim.flatten(z_lv),
                    slim.flatten(tf.zeros_like(z_mu)),
                    slim.flatten(tf.zeros_like(z_lv)),
                )
            )
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
        self.reuse=False
        #gradient penalty
        print("before gradient x shape:", x.get_shape().as_list())
        differences = xh - x
        interpolates = x + (alpha*differences)
        print("interpolates shape:", interpolates.get_shape().as_list())
        pred, inter_h = self.discriminator(interpolates, self.is_training)
        print("pred shape:", pred.get_shape().as_list())
        gradients = tf.gradients(pred, [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1,2]))
        gradient_penalty = tf.reduce_mean((slopes-1.)**2)
        disc_loss += self.arch['LAMBDA']*gradient_penalty
        self.reuse=True
        #d_loss = disc_real_loss + disc_fake_loss
        #g_loss = tf.losses.sigmoid_cross_entropy(disc_fake, tf.ones([batch_size, 1]))
        g_loss = gen_loss
        d_loss = disc_loss
        loss['xh'] = xh
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
            features = read_whole_features(self.args.file_pattern.format(self.args.src))
            x = normalizer.forward_process(features['sp'])
            x = nh_to_nchw(x)
            #y_t_id = tf.placeholder(dtype=tf.int64, shape=[1,])
            #y_t = y_t_id * tf.ones(shape=[tf.shape(x)[0],], dtype=tf.int64)
            y_t = SPEAKERS.index(self.args.trg)*tf.ones(shape=[tf.shape(x)[0],], dtype=tf.int64)
            self.reuse = False
            z, _ = self.encoder(x, False)
            x_t = self.generator(z, y_t, False)
            self.reuse = True
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from Dataset import save_images
from tfLib.ops import *
from tfLib.loss import *
from tfLib.gp import gradient_penalty
from tfLib.advloss import *
import os
import functools
import random

class SGGAN(object):

    # build model
    def __init__(self, dataset, opt):

        self.dataset = dataset
        self.opt = opt
        # placeholder
        self.x = tf.placeholder(tf.float32, [self.opt.batch_size, self.opt.img_size, self.opt.img_size, self.opt.input_nc])
        self.label = tf.placeholder(tf.float32, [self.opt.batch_size, self.opt.n_att])
        self.lr_decay = tf.placeholder(tf.float32, None, name='lr_decay')

    def build_model(self):

        self._x = self.G(self.x)
        self._x_list = tf.split(self._x, self.opt.n_att, -1)
        loss_array_d = [self.Overall_loss(item, i)[0] for i, item in enumerate(self._x_list)]
        loss_array_g = [self.Overall_loss(item, i)[1] for i, item in enumerate(self._x_list)]
        self.D_loss = tf.add_n(loss_array_d) / self.opt.n_att
        self.G_loss = tf.add_n(loss_array_g) / self.opt.n_att

    def Overall_loss(self, _x, i):

        def _t_label(j):
            label_list = tf.unstack(self.label, self.opt.n_att, axis=-1)
            return label_list[j], tf.zeros_like(label_list[0]), tf.ones_like(label_list[0])

        _x_list = tf.split(_x, 2, -1)
        #recon loss
        __x0 = self.G(_x_list[0])
        __x_list0 = tf.split(tf.split(__x0, self.opt.n_att, -1)[i], 2, -1)

        __x1 = self.G(_x_list[1])
        __x_list1 = tf.split(tf.split(__x1, self.opt.n_att, -1)[i], 2, -1)

        recon_loss = L1(_x_list[0], __x_list0[0]) + L1(_x_list[1], __x_list0[1]) + \
                     L1(_x_list[0], __x_list1[0]) + L1(_x_list[1], __x_list1[1])

        logit_gan, logits_att = self.D(self.x)
        _logit_gan_l, _logits_att_l = self.D(_x_list[0])
        _logit_gan_r, _logits_att_r = self.D(_x_list[1])

        d_att_loss = tf.losses.sigmoid_cross_entropy(_t_label(i)[0], logits_att[:,i])
        g_att_lossl = tf.losses.sigmoid_cross_entropy(_t_label(i)[1], _logits_att_l[:,i])
        g_att_lossr = tf.losses.sigmoid_cross_entropy(_t_label(i)[2], _logits_att_r[:,i])

        d_loss_fun, g_loss_fun = get_adversarial_loss(self.opt.loss_type)

        gp_lossl = gradient_penalty(lambda x: self.D(x)[0], self.x, _x_list[0], mode=self.opt.gp_type)
        gp_lossr = gradient_penalty(lambda x: self.D(x)[0], self.x, _x_list[1], mode=self.opt.gp_type)

        d_gan_loss = d_loss_fun(logit_gan, _logit_gan_l) + d_loss_fun(logit_gan, _logit_gan_r)
        g_gan_loss = g_loss_fun(_logit_gan_l) + g_loss_fun(_logit_gan_r)

        return [d_gan_loss + self.opt.lam_gp * (gp_lossl + gp_lossr) + self.opt.lam_c * d_att_loss,
                g_gan_loss + self.opt.lam_c * (g_att_lossl + g_att_lossr) + self.opt.lam_r * recon_loss]

    def train(self):

        self.t_vars = tf.trainable_variables()
        self.d_vars = [var for var in self.t_vars if 'D' in var.name]
        self.g_vars = [var for var in self.t_vars if 'G' in var.name]

        assert len(self.t_vars) == len(self.d_vars + self.g_vars)

        self.saver = tf.train.Saver()
        opti_D = tf.train.AdamOptimizer(self.opt.lr_d * self.lr_decay, beta1=self.opt.beta1, beta2=self.opt.beta2).\
                                        minimize(loss=self.D_loss, var_list=self.d_vars)
        opti_G = tf.train.AdamOptimizer(self.opt.lr_g * self.lr_decay, beta1=self.opt.beta1, beta2=self.opt.beta2).\
                                        minimize(loss=self.G_loss, var_list=self.g_vars)

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            sess.run(init)
            ckpt = tf.train.get_checkpoint_state(self.opt.checkpoints_dir)
            if ckpt and ckpt.model_checkpoint_path:
                start_step = int(ckpt.model_checkpoint_path.split('model_', 2)[1].split('.', 2)[0])
                self.saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                start_step = 0

            step = start_step
            lr_decay = 1
            print("Start read dataset")

            tr_img, tr_label, te_img, te_label = self.dataset.input()
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            _te_img, _te_label = sess.run([te_img, te_label])
            print("Start entering the looping")
            while step <= self.opt.niter:

                if step > 20000 and step % 2000 == 0:
                    lr_decay = (self.opt.niter - step) / float(self.opt.niter - 20000)

                _tr_img, _tr_label = sess.run([tr_img, tr_label])
                f_d = {self.x: _tr_img,
                       self.label: _tr_label,
                       self.lr_decay: lr_decay}
                # optimize D
                sess.run(opti_D, feed_dict=f_d)
                # optimize G
                if step % self.opt.n_critic == 0:
                    sess.run(opti_G, feed_dict=f_d)

                if step % 500 == 0:

                    o_loss = sess.run([self.D_loss, self.G_loss], feed_dict=f_d)
                    print("step %d D_loss=%.8f, G_loss=%.4f lr_decay=%.4f" %
                        (step, o_loss[0], o_loss[1], lr_decay))

                if np.mod(step, 2000) == 0:

                    tr_o = sess.run([self.x, self._x], feed_dict=f_d)

                    f_d = {self.x: _te_img,
                           self.label: _te_label}
                    te_o = sess.run([self.x, self._x], feed_dict=f_d)
                    tr_x_list = np.split(tr_o[1], indices_or_sections=self.opt.n_att * 2, axis=-1)
                    te_x_list = np.split(te_o[1], indices_or_sections=self.opt.n_att * 2, axis=-1)

                    tr_x_list.insert(0, tr_o[0])
                    te_x_list.insert(0, te_o[0])

                    _tr_o = self.Transpose(tr_x_list)
                    _te_o = self.Transpose(te_x_list)
                    save_images(_tr_o, '{}/{:02d}_tr.jpg'.format(self.opt.sample_dir, step))
                    save_images(_te_o, '{}/{:02d}_te.jpg'.format(self.opt.sample_dir, step))

                if np.mod(step, self.opt.save_model_freq) == 0:
                    self.saver.save(sess, os.path.join(self.opt.checkpoints_dir, 'model_{:06d}.ckpt'.format(step)))

                step += 1

            save_path = self.saver.save(sess, os.path.join(self.opt.checkpoints_dir, 'model_{:06d}.ckpt'.format(step)))

            coord.request_stop()
            coord.join(threads)

            print("Model saved in file: %s" % save_path)

    def Transpose(self, list):
        refined_list = np.transpose(np.array(list), axes=[1, 2, 0, 3, 4])
        refined_list = np.reshape(refined_list, [refined_list.shape[0] * refined_list.shape[1],
                                                 refined_list.shape[2] * refined_list.shape[3], -1])
        return refined_list

    def D(self, x):

        conv2d_first = functools.partial(conv2d, k=7, s=1, output_dim=self.opt.ndf, use_sp=self.opt.use_sp)
        conv2d_base = functools.partial(conv2d, use_sp=self.opt.use_sp)
        lre = functools.partial(lrelu, alpha=0.2)
        conv2d_gan = functools.partial(conv2d, k=4, s=1, output_dim=1, padding='VALID', use_sp=self.opt.use_sp)
        conv2d_att = functools.partial(conv2d, k=self.opt.img_size/pow(2, self.opt.n_layers_d),
                                       s=1, output_dim=self.opt.n_att, padding='VALID', use_sp=self.opt.use_sp)

        with tf.variable_scope("D", reuse=tf.AUTO_REUSE):

            x = lre(conv2d_first(x, scope='d_first'))
            for i in range(self.opt.n_layers_d):
                dim = np.minimum(self.opt.ndf * np.power(2, i + 1), 512)
                x = lre(conv2d_base(x, output_dim=dim, scope='d{}'.format(i)))
            logit_gan = tf.squeeze(conv2d_gan(x, scope='dgan'), axis=[1, 2])
            print(logit_gan.shape)
            logit_att = tf.squeeze(conv2d_att(x, scope='datt'), axis=[1, 2])

            return logit_gan, logit_att

    def G(self, x_init):

        conv2d_first = functools.partial(conv2d, k=3, s=1)
        conv2d_Enc = functools.partial(conv2d, k=4, s=2)
        conv2d_Dec = functools.partial(conv2d, k=4, s=1)
        conv2d_final = functools.partial(conv2d, k=7, s=1, output_dim=self.opt.output_nc)

        In = functools.partial(instance_norm)
        with tf.variable_scope("G", reuse=tf.AUTO_REUSE):

            x = x_init
            x = tf.nn.relu(In(conv2d_first(x, output_dim=self.opt.ngf, scope='conv'), scope='In'))
            for i in range(self.opt.n_layers_g):
                c_dim = np.minimum(self.opt.ngf * np.power(2, i+1), 256)
                x = tf.nn.relu(In(conv2d_Enc(x, output_dim=c_dim, scope='conv{}'.format(i)), scope='In{}'.format(i)))

            for i in range(self.opt.n_blocks):
                x = Resblock(x, o_dim=c_dim, ds=False, scope='r{}'.format(i))

            ngf = c_dim
            for i in range(self.opt.n_layers_g - 1):
                c_dim = np.maximum(int(ngf / np.power(2, i+1)), 16)
                x = tf.nn.relu(In(conv2d_Dec(x, output_dim=c_dim, scope='conv_dec{}'.format(i)), scope='de_In{}'.format(i)))
                x = upscale(x, 2)

            c_dim = c_dim // 2
            x = tf.nn.relu(In(conv2d_Dec(x, output_dim=c_dim*self.opt.n_att*2 , scope='conv_dec'), scope='de_In'))
            x = upscale(x, 2)

            x_list = tf.split(x, num_or_size_splits=self.opt.n_att*2, axis=-1)
            x_results = []
            for i in range(len(x_list)):
                x = tf.concat([x_list[i], x_init], axis=-1)
                x_results.append(conv2d_final(x, scope='f_{}'.format(i)))

            x_results = tf.concat(x_results, axis=-1)
            return tf.nn.tanh(x_results)



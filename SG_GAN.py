import tensorflow as tf
from ops import conv2d, lrelu, instance_norm, Residual, de_conv
from tensorflow.python.framework.ops import convert_to_tensor
from utils import save_images, get_image
import numpy as np
import os

class SG_GAN(object):

    # build model
    def __init__(self, batch_size, max_iters, model_path, data_ob, sample_path, log_dir, learning_rate, is_load_, use_sp,
                 lam_recon, lam_gp, range_attri, beta1, beta2):

        self.batch_size = batch_size
        self.max_iters = max_iters
        self.sg_gan_model_path = model_path
        self.data_ob = data_ob
        self.sample_path = sample_path
        self.log_dir = log_dir
        self.learning_rate = learning_rate
        self.log_vars = []
        self.channel = data_ob.channel
        self.shape = data_ob.shape
        #number of value
        self.is_load_ = is_load_
        self.use_sp = use_sp
        self.lam_recon = lam_recon
        self.lam_gp = lam_gp
        self.range_attri = range_attri
        self.beta1 = beta1
        self.beta2 = beta2
        self.output_size = data_ob.image_size

        self.y_1 = tf.placeholder(tf.int32, [batch_size, self.range_attri])
        self.y_2 = tf.placeholder(tf.int32, [batch_size, self.range_attri])
        self.y_3 = tf.placeholder(tf.int32, [batch_size, self.range_attri])
        self.y_4 = tf.placeholder(tf.int32, [batch_size, self.range_attri])
        self.images = tf.placeholder(tf.float32, [batch_size, self.output_size, self.output_size, self.channel])
        self.label_mask_1 = tf.placeholder(tf.float32, [batch_size])
        self.label_mask_2 = tf.placeholder(tf.float32, [batch_size])
        self.label_mask_hair = tf.placeholder(tf.int32, [batch_size])

        self.dataset = tf.data.Dataset.from_tensor_slices(
            (convert_to_tensor(self.data_ob.train_data_list, dtype=tf.string),
             convert_to_tensor(self.data_ob.train_lab_list1, dtype=tf.float32),
             convert_to_tensor(self.data_ob.train_lab_list2, dtype=tf.float32),
             convert_to_tensor(self.data_ob.train_lab_list3, dtype=tf.float32),
             convert_to_tensor(self.data_ob.train_lab_list4, dtype=tf.float32),
             convert_to_tensor(self.data_ob.label_mask_1, dtype=tf.float32),
             convert_to_tensor(self.data_ob.label_mask_2, dtype=tf.float32),
             convert_to_tensor(self.data_ob.label_mask_hair, dtype=tf.float32)
             ))

        self.dataset = self.dataset.shuffle(buffer_size=len(self.data_ob.train_data_list))
        self.dataset = self.dataset.map(lambda filename, label1, label2, label3, label4, mask1, mask2, mask_hair: tuple(
            tf.py_func(self._read_by_function, [filename, label1, label2, label3, label4, mask1, mask2, mask_hair], [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,
                                                                                                           tf.float32, tf.float32, tf.float32])), num_parallel_calls=32)
        self.dataset = self.dataset.repeat(50000)
        self.dataset = self.dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
        self.iterator = tf.data.Iterator.from_structure(self.dataset.output_types, self.dataset.output_shapes)
        self.next_images, self.next_label1, self.next_label2, self.next_label3, \
        self.next_label4, self.next_mask1, self.next_mask2, self.next_mask_hair = self.iterator.get_next()
        self.train_init_op = self.iterator.make_initializer(self.dataset)
        self.domain_label = tf.placeholder(tf.int32, [batch_size])
        self.lr_decay = tf.placeholder(tf.float32, None, name='lr_decay')

    def build_model_GAN(self):

        #Get the result of manipulating
        self.x_tilde_1_1, self.x_tilde_1_2, self.x_tilde_2_1, self.x_tilde_2_2, \
            self.x_tilde_3_1, self.x_tilde_3_2, self.x_tilde_4_1, self.x_tilde_4_2 = self.encode_decode(self.images, reuse=False)

        #recon_loss
        self.x_tilde_recon_1_1, _, _, _, _, _, _, _ = self.encode_decode(self.x_tilde_1_2, reuse=True)
        _, self.x_tilde_recon_1_2, _, _, _, _, _, _ = self.encode_decode(self.x_tilde_1_1, reuse=True)
        _, _, self.x_tilde_recon_2_1, _, _, _, _, _ = self.encode_decode(self.x_tilde_2_2, reuse=True)
        _, _, _, self.x_tilde_recon_2_2, _, _, _, _ = self.encode_decode(self.x_tilde_2_1, reuse=True)
        _, _, _, _, self.x_tilde_recon_3_1, _, _, _ = self.encode_decode(self.x_tilde_3_2, reuse=True)
        _, _, _, _, _, self.x_tilde_recon_3_2, _, _ = self.encode_decode(self.x_tilde_3_1, reuse=True)
        _, _, _, _, _, _, self.x_tilde_recon_4_1, _ = self.encode_decode(self.x_tilde_4_2, reuse=True)
        _, _, _, _, _, _, _, self.x_tilde_recon_4_2 = self.encode_decode(self.x_tilde_4_1, reuse=True)

        self.recon_loss = self.recon_loss(tf.concat([self.x_tilde_1_1, self.x_tilde_1_2, self.x_tilde_2_1, self.x_tilde_2_2,
                                                     self.x_tilde_3_1, self.x_tilde_3_2, self.x_tilde_4_1, self.x_tilde_4_2], axis=3),
                                          tf.concat([self.x_tilde_recon_1_1, self.x_tilde_recon_1_2, self.x_tilde_recon_2_1, self.x_tilde_recon_2_2,
                                                     self.x_tilde_recon_3_1, self.x_tilde_recon_3_2, self.x_tilde_recon_4_1, self.x_tilde_recon_4_2], axis=3))
        #classification loss
        self.D_real_class_logits_1, self.D_real_class_logits_2, self.D_real_class_logits_3, self.D_real_class_logits_4, \
            self.D_real_gan_logits = self.discriminate(self.images, reuse=False, use_sp=self.use_sp)

        self.G_fake_class_logits_1_1, _, _, _, self.D_fake_gan_logits_1_1 = self.discriminate(self.x_tilde_1_1
                                                                                              , reuse=True, use_sp=self.use_sp)
        self.G_fake_class_logits_1_2, _, _, _, self.D_fake_gan_logits_1_2 = self.discriminate(self.x_tilde_1_2
                                                                                              , reuse=True, use_sp=self.use_sp)

        _, self.G_fake_class_logits_2_1, _, _, self.D_fake_gan_logits_2_1 = self.discriminate(self.x_tilde_2_1
                                                                                              , reuse=True, use_sp=self.use_sp)
        _, self.G_fake_class_logits_2_2, _, _, self.D_fake_gan_logits_2_2 = self.discriminate(self.x_tilde_2_2
                                                                                              , reuse=True, use_sp=self.use_sp)

        _, _, self.G_fake_class_logits_3_1, _, self.D_fake_gan_logits_3_1 = self.discriminate(self.x_tilde_3_1
                                                                                              , reuse=True, use_sp=self.use_sp)
        _, _, self.G_fake_class_logits_3_2, _, self.D_fake_gan_logits_3_2 = self.discriminate(self.x_tilde_3_2
                                                                                              , reuse=True, use_sp=self.use_sp)

        _, _, _, self.G_fake_class_logits_4_1, self.D_fake_gan_logits_4_1 = self.discriminate(self.x_tilde_4_1
                                                                                              , reuse=True, use_sp=self.use_sp)
        _, _, _, self.G_fake_class_logits_4_2, self.D_fake_gan_logits_4_2 = self.discriminate(self.x_tilde_4_2
                                                                                              , reuse=True, use_sp=self.use_sp)
        #d real class loss
        self.d_class_loss_1 = self.real_class_loss(self.D_real_class_logits_1, self.y_1, self.label_mask_1)
        self.d_class_loss_2 = self.real_class_loss(self.D_real_class_logits_2, self.y_2, self.label_mask_1)
        self.d_class_loss_3 = self.real_class_loss(self.D_real_class_logits_3, self.y_3, self.label_mask_2)
        self.d_class_loss_4 = self.real_class_loss(self.D_real_class_logits_4, self.y_4, self.label_mask_1)

        # d fake class loss
        self.G_fake_class_loss_1_1, self.G_fake_class_loss_2_1, self.G_fake_class_loss_3_1, \
        self.G_fake_class_loss_4_1 = self.fake_class_loss(self.G_fake_class_logits_1_1, self.G_fake_class_logits_2_1, self.G_fake_class_logits_3_1,
                                                          self.G_fake_class_logits_4_1, tf.ones_like(self.domain_label))
        self.G_fake_class_loss_1_2, self.G_fake_class_loss_2_2, self.G_fake_class_loss_3_2, \
        self.G_fake_class_loss_4_2 = self.fake_class_loss(self.G_fake_class_logits_1_2, self.G_fake_class_logits_2_2, self.G_fake_class_logits_3_2,
                                                          self.G_fake_class_logits_4_2, tf.zeros_like(self.domain_label))

        #for attri 1
        self.d_gan_loss_1_1, self.d_gan_loss_1_2, self.d_gan_loss_2_1, self.d_gan_loss_2_2, self.d_gan_loss_3_1, \
        self.d_gan_loss_3_2, self.d_gan_loss_4_1, self.d_gan_loss_4_2 = self.d_gan_loss(self.D_fake_gan_logits_1_1, self.D_fake_gan_logits_1_2,
                                                                                        self.D_fake_gan_logits_2_1, self.D_fake_gan_logits_2_2,
                                                                                        self.D_fake_gan_logits_3_1, self.D_fake_gan_logits_3_2,
                                                                                        self.D_fake_gan_logits_4_1, self.D_fake_gan_logits_4_2,
                                                                                        self.D_real_gan_logits)

        self.d_loss_no_gp =  self.d_gan_loss_1_1 + self.d_gan_loss_1_2 + self.d_gan_loss_2_1 + self.d_gan_loss_2_2 \
                        + self.d_gan_loss_3_1 + self.d_gan_loss_3_2 + self.d_gan_loss_4_1 + self.d_gan_loss_4_2

        self.D_loss = self.d_loss_no_gp + self.d_class_loss_1 + self.d_class_loss_2 + self.d_class_loss_3 + self.d_class_loss_4

        self.all_gp = self.all_gradient_penalty(self.x_tilde_1_1, self.x_tilde_1_2, self.x_tilde_2_1, self.x_tilde_2_2, self.x_tilde_3_1, self.x_tilde_3_2,
                                                self.x_tilde_4_1, self.x_tilde_4_2, self.images)
        self.D_loss += self.lam_gp * self.all_gp

        self.G_fake_class_loss = self.G_fake_class_loss_1_1 + self.G_fake_class_loss_1_2 \
                               + self.G_fake_class_loss_2_1 + self.G_fake_class_loss_2_2 \
                               + self.G_fake_class_loss_3_1 + self.G_fake_class_loss_3_2 \
                               + self.G_fake_class_loss_4_1 + self.G_fake_class_loss_4_2

        self.g_gan_loss_1 = self.g_gan_loss(self.D_fake_gan_logits_1_1, self.D_fake_gan_logits_1_2)
        self.g_gan_loss_2 = self.g_gan_loss(self.D_fake_gan_logits_2_1, self.D_fake_gan_logits_2_2)
        self.g_gan_loss_3 = self.g_gan_loss(self.D_fake_gan_logits_3_1, self.D_fake_gan_logits_3_2)
        self.g_gan_loss_4 = self.g_gan_loss(self.D_fake_gan_logits_4_1, self.D_fake_gan_logits_4_2)

        self.g_gan_loss = self.g_gan_loss_1 + self.g_gan_loss_2 + self.g_gan_loss_3 + self.g_gan_loss_4
        self.G_loss = self.G_fake_class_loss + self.g_gan_loss + self.lam_recon * self.recon_loss

        self.t_vars = tf.trainable_variables()

        self.d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encode_decode')


        print "d_vars", len(self.d_vars)
        print "e_vars", len(self.g_vars)

        self.saver = tf.train.Saver()


    def recon_loss(self, x, x_tilde):
        return tf.reduce_mean(tf.abs(x - x_tilde))

    def real_class_loss(self, logits, lab, label_mask):

        class_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                      labels=lab)
        class_cross_entropy = tf.squeeze(class_cross_entropy)
        d_class_loss = tf.reduce_sum(label_mask * class_cross_entropy) / tf.maximum(1., tf.reduce_sum(label_mask))
        return d_class_loss

    def fake_class_loss(self, logits_1, logits_2, logits_3, logits_4, lab):

        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_1,
                                                    labels=tf.one_hot(lab, self.range_attri,
                                                                      dtype=tf.float32))), \
        tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_2,
                                                               labels=tf.one_hot(lab, self.range_attri,
                                                                                 dtype=tf.float32))), \
        tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_3,
                                                              labels=tf.one_hot(lab, self.range_attri,
                                                                                dtype=tf.float32))), \
        tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_4,
                                                              labels=tf.one_hot(lab, self.range_attri,
                                                                                dtype=tf.float32)))

    def d_gan_loss(self, fakelogit_1_1, fakelogit_1_2, fakelogit_2_1, fakelogit_2_2,
                   fakelogit_3_1, fakelogit_3_2, fakelogit_4_1, fakelogit_4_2, reallogit):
        return tf.reduce_mean(fakelogit_1_1) - tf.reduce_mean(reallogit), \
               tf.reduce_mean(fakelogit_1_2) - tf.reduce_mean(reallogit), \
               tf.reduce_mean(fakelogit_2_1) - tf.reduce_mean(reallogit), \
               tf.reduce_mean(fakelogit_2_2) - tf.reduce_mean(reallogit), \
               tf.reduce_mean(fakelogit_3_1) - tf.reduce_mean(reallogit), \
               tf.reduce_mean(fakelogit_3_2) - tf.reduce_mean(reallogit), \
               tf.reduce_mean(fakelogit_4_1) - tf.reduce_mean(reallogit), \
               tf.reduce_mean(fakelogit_4_2) - tf.reduce_mean(reallogit)

    def g_gan_loss(self, fake_logit_1, fake_logit_2):
        return - tf.reduce_mean(fake_logit_1) - tf.reduce_mean(fake_logit_2)

    def gradient_penalty(self, x_tilde, x):

        self.differences = x_tilde - x
        self.alpha = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0., maxval=1.)
        interpolates = x + self.alpha * self.differences
        _, _, _, _, discri_logits = self.discriminate(interpolates, reuse=True)
        gradients = tf.gradients(discri_logits, [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2, 3]))
        return tf.reduce_mean((slopes - 1.)**2)

    def all_gradient_penalty(self, x_tilde_1_1, x_tilde_1_2, x_tilde_2_1, x_tilde_2_2, x_tilde_3_1, x_tilde_3_2, x_tilde_4_1, x_tilde_4_2, x):

        return self.gradient_penalty(x_tilde_1_1, x) + self.gradient_penalty(x_tilde_1_2, x) + self.gradient_penalty(x_tilde_2_1, x) + \
               self.gradient_penalty(x_tilde_2_2, x) + self.gradient_penalty(x_tilde_3_1, x) + self.gradient_penalty(
            x_tilde_3_2, x) + self.gradient_penalty(x_tilde_4_1, x) + self.gradient_penalty(x_tilde_4_2, x)

    def _read_by_function(self, filename, label1, label2, label3, label4, mask1, mask2, mask_hair):

        array = get_image(filename, self.output_size, is_crop=True, resize_w=self.output_size,
                          is_grayscale=False)
        real_images = np.array(array, dtype=np.float32)
        return real_images, label1, label2, label3, label4, mask1, mask2, mask_hair

    def test_class_accuracy(self, session, test_data1, test_data2):

        correct_1 = 0
        correct_2 = 0
        correct_3 = 0
        correct_4 = 0

        iters = len(test_data1) / self.batch_size
        for i in range(iters):

            test_data_list, test_lab_list1, test_lab_list2, _, test_lab_list4, _ = self.data_ob.getTestNextBatch(i,
                                                                                                                 self.batch_size,
                                                                                                                 is_shuffle=False)
            batch_images_array = self.data_ob.getShapeForData(test_data_list)
            class_logits_on_data_1, class_logits_on_data_2, class_logits_on_data_4 = \
                session.run([self.D_real_class_logits_1, self.D_real_class_logits_2, self.D_real_class_logits_4],
                         feed_dict={self.images: batch_images_array})

            pred_class = np.argmax(np.int32(class_logits_on_data_1), 1)
            eq = np.equal(np.int32(test_lab_list1), pred_class)
            correct_1 += np.sum(eq)

            pred_class = np.argmax(np.int32(class_logits_on_data_2), 1)
            eq = np.equal(np.int32(test_lab_list2), pred_class)
            correct_2 += np.sum(eq)

            pred_class = np.argmax(np.int32(class_logits_on_data_4), 1)
            eq = np.equal(np.int32(test_lab_list4), pred_class)
            correct_4 += np.sum(eq)

        iters2 = len(test_data2) / self.batch_size

        for i in range(iters2):

            test_data_list, test_lab_list = self.data_ob.getTest_forhairNextBatch(i, self.batch_size, is_shuffle=False)
            batch_images_array = self.data_ob.getShapeForData(test_data_list)
            class_logits_on_data_3 = session.run(self.D_real_class_logits_3, feed_dict={self.images: batch_images_array})

            pred_class = np.argmax(np.int32(class_logits_on_data_3), 1)
            eq = np.equal(np.int32(test_lab_list), pred_class)

            correct_3 += np.sum(eq)

        print "Test Accuracy", "Gender: %.3f Smile: %.3f Hair Color: %.3f Lipstick: %.3f" % (correct_1 / (np.float(iters * self.batch_size)), correct_2 / (np.float(iters * self.batch_size)), correct_3 / (np.float(iters2 * self.batch_size))\
            , correct_4 / (np.float(iters * self.batch_size)))

    def test(self, test_step):

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            sess.run(init)

            self.saver.restore(sess, os.path.join(self.sg_gan_model_path, 'model_{:06d}.ckpt'.format(test_step)))
            batch_num = len(self.data_ob.test_data_list) / self.batch_size

            for j in range(batch_num):

                train_data_list, _, _, _, _, _ = self.data_ob.getTestNextBatch(batch_num=j, batch_size=self.batch_size, is_shuffle=False)
                batch_images_array = self.data_ob.getShapeForData(train_data_list, is_test=True)

                x1_1, x1_2, x2_1, x2_2, x3_1, x3_1, x4_1, x4_2 = sess.run(
                    [self.x_tilde_1_1, self.x_tilde_1_2,
                     self.x_tilde_2_1, self.x_tilde_2_2, self.x_tilde_3_1, self.x_tilde_3_2, self.x_tilde_4_1, self.x_tilde_4_2],
                    feed_dict={self.images: batch_images_array})

                for i in range(self.batch_size):

                    output_concat = np.concatenate([batch_images_array, x1_1, x1_2, x2_1, x2_2, x3_1, x3_1,
                                                    x4_1, x4_2], axis=0)
                    save_images(output_concat, [output_concat.shape[0] / 8, 8],
                                '{}/{:02d}_output.jpg'.format(self.sample_path, i))

    # do train
    def train(self):

        opti_D = tf.train.AdamOptimizer(self.learning_rate * self.lr_decay, beta1=self.beta1, beta2=self.beta2).minimize(self.D_loss, var_list=self.d_vars)
        opti_M = tf.train.AdamOptimizer(self.learning_rate * self.lr_decay, beta1=self.beta1, beta2=self.beta2).minimize(self.G_loss, var_list=self.g_vars)

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            sess.run(init)
            sess.run(self.train_init_op)
            step = 0
            step_2 = 0

            if self.is_load_:
                self.saver.restore(sess, self.sg_gan_model_path + str(step))
            lr_decay = 1
            while step <= self.max_iters:

                if step > 20000 and lr_decay > 0.1:
                    lr_decay = (self.max_iters - step) / float(self.max_iters - 20000)

                # optimization D
                for i in range(5):

                    batch_image_array, batch_label1, batch_label2, batch_label3, batch_label4, \
                            batch_mask1, batch_mask2, batch_mask_hair \
                        = sess.run([self.next_images, self.next_label1, self.next_label2, self.next_label3, self.next_label4,
                              self.next_mask1, self.next_mask2, self.next_mask_hair])

                    f_d = {self.images: batch_image_array, self.y_1: batch_label1, self.y_2: batch_label2,
                          self.y_3: batch_label3, self.y_4: batch_label4,
                         self.label_mask_1: batch_mask1, self.label_mask_2: batch_mask2,
                           self.label_mask_hair: batch_mask_hair, self.lr_decay: lr_decay}

                    sess.run(opti_D, feed_dict= f_d)
                    step_2 += 1

                sess.run(opti_M, feed_dict= f_d)

                if step % 200 == 0:

                    d_loss, d_loss_no_gp, d_gan_loss_1, d_gan_loss_2, d_gan_loss_3, d_gan_loss_4, g_loss, g_class_loss, g_gan_loss, \
                        g_gan_loss_1, g_gan_loss_2, g_gan_loss_3, g_gan_loss_4, recon_loss \
                        = sess.run([self.D_loss, self.d_loss_no_gp, self.d_gan_loss_1_1 + self.d_gan_loss_1_2,
                            self.d_gan_loss_2_1 + self.d_gan_loss_2_2, self.d_gan_loss_3_1 + self.d_gan_loss_3_2,
                            self.d_gan_loss_4_1 + self.d_gan_loss_4_2, self.G_loss, self.G_fake_class_loss,
                                    self.g_gan_loss, self.g_gan_loss_1, self.g_gan_loss_2, self.g_gan_loss_3, self.g_gan_loss_4, self.recon_loss],
                        feed_dict= f_d)

                    print("step %d D_loss =%.3f, D_loss_no_gp=%.3f, D_gan_loss_1=%.3f, D_gan_loss_2=%.3f, D_gan_loss_3=%.3f, D_gan_loss_4=%.3f \n"
                          "G_loss=%.3f, G_class_loss=%.3f, G_gan_loss=%.3f, G_gan_loss_1=%.3f, G_gan_loss_2=%.3f, G_gan_loss_3=%.3f, G_gan_loss_4=%.3f, Recon_loss= %.3f \n"
                          "label_mask_1=%i" % (step, d_loss, d_loss_no_gp, d_gan_loss_1, d_gan_loss_2, d_gan_loss_3, d_gan_loss_4, g_loss, g_class_loss, g_gan_loss,
                                    g_gan_loss_1, g_gan_loss_2, g_gan_loss_3, g_gan_loss_4, recon_loss, np.sum(batch_mask1, dtype=np.int32)))

                if np.mod(step, 400) == 0:

                    test_data_list, _, _, _, _, _ = self.data_ob.getTestNextBatch(0, self.batch_size, is_shuffle=False)
                    batch_images_array = self.data_ob.getShapeForData(test_data_list, is_test=True)
                    f_d = {self.images: batch_images_array}

                    x_tilde_1_1, x_tilde_1_2, x_tilde_2_1, x_tilde_2_2, x_tilde_3_1, x_tilde_3_2, x_tilde_4_1, x_tilde_4_2 \
                        = sess.run([self.x_tilde_1_1, self.x_tilde_1_2, self.x_tilde_2_1,
                                    self.x_tilde_2_2, self.x_tilde_3_1, self.x_tilde_3_2, self.x_tilde_4_1, self.x_tilde_4_2], feed_dict= f_d)

                    output_concat = np.concatenate([batch_images_array, x_tilde_1_1,
                                                    x_tilde_1_2, x_tilde_2_1, x_tilde_2_2, x_tilde_3_1, x_tilde_3_2,
                                                    x_tilde_4_1, x_tilde_4_2], axis=0)
                    # gender
                    save_images(output_concat, [output_concat.shape[0]/8, 8], '{}/{:02d}_output.jpg'.format(self.sample_path, step))

                if np.mod(step, 2000) == 0 and step != 0:
                    self.saver.save(sess, os.path.join(self.sg_gan_model_path, 'model_{:06d}.ckpt'.format(step)))

                # Test
                if np.mod(step, 1000) == 0:
                    self.test_class_accuracy(sess, self.data_ob.test_data_list, self.data_ob.test_data_list_forhair)
                step += 1

            save_path = self.saver.save(sess, os.path.join(self.sg_gan_model_path, 'model_{:06d}.ckpt'.format(step)))
            print "Model saved in file: %s" % save_path

    def discriminate(self, x_var, sn=64, reuse=False, use_sp=False):

        with tf.variable_scope("discriminator") as scope:

            if reuse == True:
                scope.reuse_variables()

            conv1= lrelu(conv2d(x_var, spectural_normed=use_sp, output_dim=sn, name='dis_conv1'))
            conv2= lrelu(conv2d(conv1, spectural_normed=use_sp, output_dim=sn*2, name='dis_conv2'))
            conv3= lrelu(conv2d(conv2, spectural_normed=use_sp, output_dim=sn*4, name='dis_conv3'))
            conv4 = lrelu(conv2d(conv3, spectural_normed=use_sp, output_dim=sn*8, name='dis_conv4'))
            conv5 = lrelu(conv2d(conv4, spectural_normed=use_sp, output_dim=sn*8, name='dis_conv5'))
            conv6 = lrelu(conv2d(conv5, spectural_normed=use_sp, output_dim=sn*16, name='dis_conv6'))

            #for gender
            class_logits_1 = conv2d(conv6, spectural_normed=use_sp, output_dim=self.range_attri, k_h=conv6.shape[1],
                                    k_w=conv6.shape[1], d_w=1, d_h=1, padding='VALID', name='dis_conv7')
            #for smile
            class_logits_2 = conv2d(conv6, spectural_normed=use_sp, output_dim=self.range_attri, k_h=conv6.shape[1],
                                    k_w=conv6.shape[1], d_w=1, d_h=1, padding='VALID', name='dis_conv8')

            #for hair color
            class_logits_3 = conv2d(conv6, spectural_normed=use_sp, output_dim=self.range_attri, k_h=conv6.shape[1],
                                    k_w=conv6.shape[1], d_w=1, d_h=1, padding='VALID', name='dis_conv9')
            # for lipsticks
            class_logits_4 = conv2d(conv6, spectural_normed=use_sp, output_dim=self.range_attri, k_h=conv6.shape[1],
                                    k_w=conv6.shape[1], d_w=1, d_h=1, padding='VALID', name='dis_conv10')

            #PatchGAN
            gan_logits = conv2d(conv6, spectural_normed=use_sp, output_dim=1, k_h=1, k_w=1, d_w=1, d_h=1,
                                padding='VALID', name='dis_conv11')

            return tf.squeeze(class_logits_1), tf.squeeze(class_logits_2), tf.squeeze(class_logits_3), tf.squeeze(class_logits_4), \
                        tf.squeeze(gan_logits)

    def encode_decode(self, x, sn=64, reuse=False):

        print sn

        with tf.variable_scope("encode_decode") as scope:

            if reuse == True:
                scope.reuse_variables()

            conv1 = tf.nn.relu(
                instance_norm(conv2d(x, output_dim=sn, k_w=7, k_h=7, d_w=1, d_h=1, name='e_c1'), scope='e_in1'))
            conv2 = tf.nn.relu(
                instance_norm(conv2d(conv1, output_dim=sn*2, k_w=4, k_h=4, d_w=2, d_h=2, name='e_c2'), scope='e_in2'))
            conv3 = tf.nn.relu(
                instance_norm(conv2d(conv2, output_dim=sn*4, k_w=4, k_h=4, d_w=2, d_h=2, name='e_c3'), scope='e_in3'))

            r1 = Residual(conv3, residual_name='re_1')
            r2 = Residual(r1, residual_name='re_2')
            r3 = Residual(r2, residual_name='re_3')
            r4 = Residual(r3, residual_name='re_4')
            r5 = Residual(r4, residual_name='re_5')
            r6 = Residual(r5, residual_name='re_6')

            g_deconv1 = tf.nn.relu(instance_norm(de_conv(r6, output_shape=[self.batch_size,
                                                                           self.output_size/2, self.output_size/2, sn*2], name='gen_deconv1'), scope="gen_in"))
            # for 1
            g_deconv_1_1 = tf.nn.relu(instance_norm(de_conv(g_deconv1,
                        output_shape=[self.batch_size, self.output_size, self.output_size, sn], name='g_deconv_1_1'), scope='gen_in_1_1'))

            #Refined Residual Image learning
            g_deconv_1_1_x = tf.concat([g_deconv_1_1, x], axis=3)
            x_tilde1 = conv2d(g_deconv_1_1_x, output_dim=self.channel, k_w=7, k_h=7, d_h=1, d_w=1, name='gen_conv_1_2')

            # for 2
            g_deconv_2_1 = tf.nn.relu(instance_norm(de_conv(g_deconv1,
                        output_shape=[self.batch_size, self.output_size, self.output_size, sn]
                                                            , name='g_deconv_2_1'), scope='gen_in_2_1'))
            g_deconv_2_1_x = tf.concat([g_deconv_2_1, x], axis=3)
            x_tilde2 = conv2d(g_deconv_2_1_x, output_dim=self.channel, k_w=7, k_h=7, d_h=1, d_w=1, name='gen_conv_2_2')

            # for 3
            g_deconv_3_1 = tf.nn.relu(instance_norm(de_conv(g_deconv1, output_shape=[self.batch_size,
                                            self.output_size, self.output_size, sn], name='gen_deconv3_1'), scope='gen_in_3_1'))
            g_deconv_3_1_x = tf.concat([g_deconv_3_1, x], axis=3)
            g_deconv_3_2 = conv2d(g_deconv_3_1_x, output_dim=32, k_w=3, k_h=3, d_h=1, d_w=1,
                              name='gen_conv_3_2')
            x_tilde3 = conv2d(g_deconv_3_2, output_dim=3, k_h=3, k_w=3, d_h=1, d_w=1, name='gen_conv_3_3')

            # for 4
            g_deconv_4_1 = tf.nn.relu(instance_norm(de_conv(g_deconv1, output_shape=[self.batch_size,
                                                                        self.output_size, self.output_size, sn], name='gen_deconv4_1'), scope='gen_in_4_1'))
            g_deconv_4_1_x = tf.concat([g_deconv_4_1, x], axis=3)
            g_deconv_4_2 = conv2d(g_deconv_4_1_x, output_dim=32, k_w=3, k_h=3, d_h=1, d_w=1,
                              name='gen_conv_4_2')
            x_tilde4 = conv2d(g_deconv_4_2, output_dim=3, k_h=3, k_w=3, d_h=1, d_w=1, name='gen_conv_4_3')

            # for 5
            g_deconv_5_1 = tf.nn.relu(instance_norm(de_conv(g_deconv1, output_shape=[self.batch_size,
                                                                        self.output_size, self.output_size, sn], name='gen_deconv5_1'), scope='gen_in_5_1'))
            g_deconv_5_1_x = tf.concat([g_deconv_5_1, x], axis=3)
            g_deconv_5_2 = conv2d(g_deconv_5_1_x, output_dim=32, k_w=3, k_h=3, d_h=1, d_w=1,
                              name='gen_conv_5_2')
            x_tilde5 = conv2d(g_deconv_5_2, output_dim=3, k_h=3, k_w=3, d_h=1, d_w=1, name='gen_conv_5_3')

            # for 6
            g_deconv_6_1 = tf.nn.relu(instance_norm(de_conv(g_deconv1, output_shape=[self.batch_size,
                                                                        self.output_size, self.output_size, sn], name='gen_deconv6_1'), scope='gen_in_6_1'))
            g_deconv_6_1_x = tf.concat([g_deconv_6_1, x], axis=3)
            g_deconv_6_2 = conv2d(g_deconv_6_1_x, output_dim=32, k_w=3, k_h=3, d_h=1, d_w=1,
                              name='gen_conv_6_2')
            x_tilde6 = conv2d(g_deconv_6_2, output_dim=3, k_h=3, k_w=3, d_h=1, d_w=1, name='gen_conv_6_3')

            # for 7
            g_deconv_7_1 = tf.nn.relu(instance_norm(de_conv(g_deconv1,
                        output_shape=[self.batch_size, self.output_size, self.output_size, sn], name='g_deconv_7_1'), scope='gen_in_7_1'))

            g_deconv_7_1_x = tf.concat([g_deconv_7_1, x], axis=3)
            x_tilde7 = conv2d(g_deconv_7_1_x, output_dim=self.channel, k_w=7, k_h=7, d_h=1, d_w=1, name='gen_conv_7_2')

            # for 8
            g_deconv_8_1 = tf.nn.relu(instance_norm(de_conv(g_deconv1,
                        output_shape=[self.batch_size, self.output_size, self.output_size, sn]
                                                            , name='g_deconv_8_1'), scope='gen_in_8_1'))
            g_deconv_8_1_x = tf.concat([g_deconv_8_1, x], axis=3)
            x_tilde8 = conv2d(g_deconv_8_1_x, output_dim=self.channel, k_w=7, k_h=7, d_h=1, d_w=1, name='gen_conv_8_2')

            return tf.nn.tanh(x_tilde1), tf.nn.tanh(x_tilde2), tf.nn.tanh(x_tilde3), \
                   tf.nn.tanh(x_tilde4), tf.nn.tanh(x_tilde5), tf.nn.tanh(x_tilde6), tf.nn.tanh(x_tilde7), tf.nn.tanh(x_tilde8)


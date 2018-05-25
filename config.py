import os
import tensorflow as tf

class Config:

    @property
    def base_dir(self):
        return os.path.abspath(os.curdir)

    @property
    def data_dir(self):
        data_dir = os.path.join(self.base_dir, '/home/wangbin/data/')
        if not os.path.exists(data_dir):
            raise ValueError('Please specify a data dir.')
        return data_dir

    @property
    def exp_dir(self):
        exp_dir = os.path.join(self.base_dir, 'train_log_' + str(Config.OPER_NAME))
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
        return exp_dir

    @property
    def model_path(self):
        model_path = os.path.join(self.exp_dir, 'model')
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        return model_path

    @property
    def log_dir(self):
        log_dir = os.path.join(self.exp_dir, 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    @property
    def sample_path(self):
        sample_path = os.path.join(self.exp_dir, 'sample_img_' + str(Config.OPER_FLAG))
        if not os.path.exists(sample_path):
            os.makedirs(sample_path)
        return sample_path

    def get_hyper_para(self):

        flags = tf.app.flags
        flags.DEFINE_integer("batch_size", 8, "size of single batch")
        flags.DEFINE_integer("learn_rate", 0.0001, "learning rate for g and d")
        flags.DEFINE_integer("image_size", 128, "size of image for training and testing")
        flags.DEFINE_integer("max_iters", 40000, "number of total iterations for G")
        flags.DEFINE_integer("beta1", 0.5, "beta1 of Adam")
        flags.DEFINE_integer("beta2", 0.999, "beta2 of Adam")
        flags.DEFINE_integer("test_step", 40000, "step in test")
        flags.DEFINE_integer("range_attri", 2, "range of attribute value")
        flags.DEFINE_integer("sp_type", 0, "type of sparsely grouped")
        flags.DEFINE_boolean("is_load", False, "whether loading the pretraining model")
        flags.DEFINE_boolean("use_sp", False, "whether using spectral normalization")
        flags.DEFINE_integer("lam_recon", 10, "weight for recon_loss")

        FLAG = flags.FLAGS
        return FLAG

    OPER_FLAG = 0
    OPER_NAME = "experiment_5_23"
    gpu_id = 15
    image_size = 128
    channel = 3
    hwc = [image_size, image_size, channel]
    select_attribute = []


from utils import CelebA
from SG_GAN import SG_GAN
from config import Config
import os

config = Config()
os.environ['CUDA_VISIBLE_DEVICES']=str(config.gpu_id)

FLAGS = config.get_hyper_para()

if __name__ == "__main__":

    m_ob = CelebA(config.data_dir, FLAGS.image_size, FLAGS.sp_type)
    sg_gan = SG_GAN(batch_size= FLAGS.batch_size, max_iters= FLAGS.max_iters,
                      model_path= config.model_path, data_ob= m_ob,
                      sample_path= config.sample_path , log_dir= config.exp_dir,
                      learning_rate= FLAGS.learn_rate, is_load_= FLAGS.is_load, use_sp=FLAGS.use_sp,
                      lam_recon= FLAGS.lam_recon, lam_gp= FLAGS.lam_gp, range_attri= FLAGS.range_attri, beta1= FLAGS.beta1, beta2= FLAGS.beta2)

    if config.OPER_FLAG == 0:

        sg_gan.build_model_GAN()
        sg_gan.train()

    if config.OPER_FLAG == 1:

        sg_gan.build_model_GAN()
        sg_gan.test(test_step=FLAGS.test_step)





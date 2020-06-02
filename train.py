from Dataset import CelebA
from SGGAN import SGGAN
import os
from config.train_options import TrainOptions

opt = TrainOptions().parse()
os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_id)

if __name__ == "__main__":

    m_ob = CelebA(opt)
    sggan = SGGAN(m_ob, opt=opt)
    sggan.build_model()
    sggan.train()

